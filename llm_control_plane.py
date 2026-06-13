import sys
import threading

import uvicorn

from src.orchestrator.config import MISSING_CONFIG_MESSAGE, require_config_file


def run_proxy():
    uvicorn.run("src.orchestrator.proxy:app", host="0.0.0.0", port=12340, reload=False)


def run_dashboard():
    from src.dashboard.app import app

    app.run(host="127.0.0.1", port=12341)


def main() -> int:
    try:
        require_config_file()
    except RuntimeError:
        print(MISSING_CONFIG_MESSAGE, file=sys.stderr)
        return 1

    t1 = threading.Thread(target=run_proxy, daemon=True)
    t2 = threading.Thread(target=run_dashboard, daemon=True)
    t2.start()
    t1.start()
    t1.join()
    t2.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
