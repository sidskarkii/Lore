"""TutorialVault server — run with: python -m tutorialvault.server"""

import uvicorn

from .api.app import create_app

app = create_app()


def main(host: str = "127.0.0.1", port: int = 8000):
    print(f"\n  TutorialVault API starting on http://{host}:{port}")
    print(f"  Docs: http://{host}:{port}/api/docs")
    print(f"  Health: http://{host}:{port}/api/health\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
