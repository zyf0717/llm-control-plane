import threading

import uvicorn

from src.dashboard.app import app


def run_proxy():
    uvicorn.run("src.orchestrator.proxy:app", host="0.0.0.0", port=12340, reload=False)


def run_dashboard():
    app.run(host="0.0.0.0", port=12341)


if __name__ == "__main__":
    t1 = threading.Thread(target=run_proxy, daemon=True)
    t2 = threading.Thread(target=run_dashboard, daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
