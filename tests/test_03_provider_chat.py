"""Test provider chat — actually send a message through each available provider."""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from lore.providers.registry import ProviderRegistry

TEST_MESSAGES = [{"role": "user", "content": "Say 'hello' in exactly one word. Nothing else."}]


def test_provider_chat():
    registry = ProviderRegistry()
    all_status = registry.all_status()

    results = {}

    for name, info in all_status.items():
        if not info["installed"]:
            print(f"  {name}: SKIPPED (not installed)")
            results[name] = {"status": "skipped", "reason": "not installed"}
            continue

        if name == "custom" and not info["authenticated"]:
            print(f"  {name}: SKIPPED (not configured)")
            results[name] = {"status": "skipped", "reason": "not configured"}
            continue

        provider = registry.get(name)

        # Pick a free model if available, else first model
        model = None
        free = [m["id"] for m in info["models"] if m["free"]]
        if free:
            model = free[0]
        elif info["models"]:
            model = info["models"][0]["id"]

        print(f"\n--- {info['display_name']} (model: {model}) ---")

        try:
            t0 = time.time()
            response = provider.chat(TEST_MESSAGES, model=model)
            latency = time.time() - t0

            print(f"  Response: {response[:200]}")
            print(f"  Latency: {latency:.1f}s")
            print(f"  Length: {len(response)} chars")

            results[name] = {
                "status": "ok",
                "response": response[:200],
                "latency_s": round(latency, 1),
                "model": model,
            }
        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = {"status": "error", "error": str(e)[:200]}

    # Summary
    print(f"\n{'='*60}")
    print("  CHAT TEST SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        status = r["status"]
        if status == "ok":
            print(f"  {name:<15} OK  ({r['latency_s']}s) \"{r['response'][:50]}\"")
        elif status == "skipped":
            print(f"  {name:<15} SKIP ({r['reason']})")
        else:
            print(f"  {name:<15} FAIL ({r['error'][:60]})")


if __name__ == "__main__":
    test_provider_chat()
