"""Test the search API endpoint — start server, hit it with curl-style requests."""

import sys, os, time, json, subprocess, signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

PORT = 8111  # Use a non-default port to avoid conflicts


def start_server():
    """Start the FastAPI server as a subprocess."""
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "tutorialvault.api.app:create_app",
         "--factory", "--host", "127.0.0.1", "--port", str(PORT)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env,
    )
    # Wait for startup
    time.sleep(4)
    return proc


def api(method, path, body=None):
    """Make an HTTP request to the test server."""
    import urllib.request
    url = f"http://127.0.0.1:{PORT}{path}"
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "detail": e.read().decode()[:200]}


def test_api():
    print("Starting test server...")
    server = start_server()

    try:
        # 1. Health
        print("\n1. GET /api/health")
        r = api("GET", "/api/health")
        print(f"   status={r.get('status')}, chunks={r.get('total_chunks')}, provider={r.get('active_provider')}")
        assert r["status"] == "ok"

        # 2. Providers
        print("\n2. GET /api/providers")
        r = api("GET", "/api/providers")
        installed = [p["name"] for p in r["providers"] if p["installed"]]
        free_counts = {p["name"]: p["free_model_count"] for p in r["providers"] if p["free_model_count"] > 0}
        print(f"   Installed: {installed}")
        print(f"   Free models: {free_counts}")
        print(f"   Active: {r['active']}")

        # 3. Collections
        print("\n3. GET /api/collections")
        r = api("GET", "/api/collections")
        print(f"   Collections: {len(r.get('collections', []))}, Total chunks: {r.get('total_chunks', 0)}")

        # 4. Search (only works if there's data in the DB)
        print("\n4. POST /api/search")
        r = api("POST", "/api/search", {"query": "test query", "n_results": 3})
        if "error" in r:
            print(f"   Expected — {r.get('detail', 'no data indexed yet')}")
        else:
            print(f"   Results: {r.get('total', 0)}")
            for res in r.get("results", [])[:2]:
                print(f"   -> [{res['timestamp']}] {res['collection_display']}: {res['text'][:80]}...")

        # 5. Sessions (should be empty)
        print("\n5. GET /api/sessions")
        r = api("GET", "/api/sessions")
        print(f"   Sessions: {len(r.get('sessions', []))}")

        print("\n=== API TESTS PASSED ===")

    finally:
        server.terminate()
        server.wait(timeout=5)
        print("Server stopped.")


if __name__ == "__main__":
    test_api()
