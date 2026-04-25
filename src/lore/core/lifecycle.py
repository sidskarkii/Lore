"""Model lifecycle manager -- JIT loading with TTL-based eviction.

Keeps heavy ML models (embedding, reranker, STT, NER) loaded only while
actively used. A background sweep every 5 minutes evicts idle models,
freeing multi-GB of RAM between work phases.

Each model is a Slot with:
  - Lazy loading on first acquire()
  - Reference counting so eviction never races active use
  - TTL measured from last release, not last access
"""

from __future__ import annotations

import atexit
import gc
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable

_SWEEP_INTERVAL = 300  # 5 minutes


class Slot:
    """A managed model with refcounted access and TTL eviction."""

    __slots__ = (
        "name", "loader", "cleanup", "ttl", "ram_mb",
        "_instance", "_active", "_last_released", "_lock",
    )

    def __init__(
        self,
        name: str,
        loader: Callable[[], Any],
        cleanup: Callable[[Any], None] | None = None,
        ttl: float = 300,
        ram_mb: int = 0,
    ):
        self.name = name
        self.loader = loader
        self.cleanup = cleanup
        self.ttl = ttl
        self.ram_mb = ram_mb
        self._instance: Any = None
        self._active: int = 0
        self._last_released: float = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> Any:
        """Load if needed, increment refcount, return instance."""
        with self._lock:
            if self._instance is None:
                t0 = time.monotonic()
                self._instance = self.loader()
                dt = time.monotonic() - t0
                print(f"  [lifecycle] {self.name} loaded ({dt:.1f}s)")
            self._active += 1
            return self._instance

    def release(self):
        """Decrement refcount. Eviction-eligible when count hits 0."""
        with self._lock:
            self._active = max(0, self._active - 1)
            if self._active == 0:
                self._last_released = time.monotonic()

    @contextmanager
    def lease(self):
        """Context manager for safe acquire/release."""
        instance = self.acquire()
        try:
            yield instance
        finally:
            self.release()

    @property
    def loaded(self) -> bool:
        return self._instance is not None

    @property
    def idle(self) -> float:
        if not self.loaded or self._active > 0:
            return 0.0
        return time.monotonic() - self._last_released

    @property
    def evictable(self) -> bool:
        return (
            self._instance is not None
            and self._active == 0
            and self.idle > self.ttl
        )

    def evict(self) -> bool:
        """Evict if idle longer than TTL and no active users."""
        with self._lock:
            if self._instance is None or self._active > 0:
                return False
            if time.monotonic() - self._last_released < self.ttl:
                return False
            if self.cleanup:
                try:
                    self.cleanup(self._instance)
                except Exception:
                    pass
            self._instance = None
            print(f"  [lifecycle] {self.name} evicted (~{self.ram_mb}MB freed)")
            return True


class ModelManager:
    """Singleton. Registers model slots, sweeps idle ones every 5 minutes."""

    _instance: "ModelManager | None" = None
    _init_lock = threading.Lock()

    def __init__(self):
        self._slots: dict[str, Slot] = {}
        self._slots_lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._running = False

    @classmethod
    def get(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = ModelManager()
                    atexit.register(cls._instance.shutdown)
        return cls._instance

    def register(
        self,
        name: str,
        loader: Callable[[], Any],
        cleanup: Callable[[Any], None] | None = None,
        ttl: float = 300,
        ram_mb: int = 0,
    ) -> Slot:
        """Register a model slot. Idempotent -- returns existing slot if already registered."""
        with self._slots_lock:
            if name in self._slots:
                return self._slots[name]
            slot = Slot(name, loader, cleanup, ttl, ram_mb)
            self._slots[name] = slot
        self._ensure_sweep()
        return slot

    def status(self) -> list[dict]:
        """Snapshot of all slots for diagnostics."""
        with self._slots_lock:
            slots = list(self._slots.values())
        return [
            {
                "name": s.name,
                "loaded": s.loaded,
                "active": s._active,
                "idle_s": round(s.idle, 1) if s.loaded else None,
                "ttl_s": int(s.ttl),
                "ram_mb": s.ram_mb if s.loaded else 0,
            }
            for s in slots
        ]

    def loaded_ram_mb(self) -> int:
        """Total estimated RAM of all loaded models."""
        with self._slots_lock:
            slots = list(self._slots.values())
        return sum(s.ram_mb for s in slots if s.loaded)

    def _ensure_sweep(self):
        if not self._running:
            self._running = True
            self._schedule()

    def _schedule(self):
        self._timer = threading.Timer(_SWEEP_INTERVAL, self._sweep)
        self._timer.daemon = True
        self._timer.start()

    def _sweep(self):
        with self._slots_lock:
            slots = list(self._slots.values())
        evicted = False
        for slot in slots:
            if slot.evict():
                evicted = True
        if evicted:
            gc.collect()
        if self._running:
            self._schedule()

    def unload_all(self):
        """Force-unload everything, ignoring TTL and refcounts."""
        with self._slots_lock:
            slots = list(self._slots.values())
        for slot in slots:
            with slot._lock:
                if slot._instance is not None:
                    if slot.cleanup:
                        try:
                            slot.cleanup(slot._instance)
                        except Exception:
                            pass
                    slot._instance = None
        gc.collect()

    def shutdown(self):
        self._running = False
        if self._timer:
            self._timer.cancel()
        self.unload_all()


def get_model_manager() -> ModelManager:
    """Module-level accessor for the singleton."""
    return ModelManager.get()
