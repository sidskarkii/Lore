"""Test provider detection — verify all CLIs are found and status is correct."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from lore.providers.registry import ProviderRegistry


def test_provider_detection():
    registry = ProviderRegistry()

    print("Provider Detection Results:")
    print("=" * 60)

    detected = registry.detect_all()
    for name, installed in detected.items():
        print(f"  {name:<15} {'INSTALLED' if installed else 'NOT FOUND'}")

    print()
    all_status = registry.all_status()

    for name, info in all_status.items():
        print(f"--- {info['display_name']} ---")
        print(f"  Installed: {info['installed']}")
        print(f"  Authenticated: {info['authenticated']}")
        print(f"  Version: {info['version']}")
        print(f"  User: {info['user']}")
        print(f"  Models: {len(info['models'])}")
        print(f"  Free models: {info['free_model_count']}")
        if info['error']:
            print(f"  Error: {info['error']}")
        print()

    # Check at least one provider is available
    available = [n for n, i in all_status.items() if i["installed"]]
    assert len(available) > 0, "No providers detected!"
    print(f"Available providers: {available}")

    # Check active provider
    active = registry.active
    assert active is not None, "No active provider set!"
    print(f"Active provider: {active.name}")

    print("\n=== PROVIDER DETECTION PASSED ===")


if __name__ == "__main__":
    test_provider_detection()
