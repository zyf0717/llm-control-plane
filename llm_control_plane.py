import threading

import uvicorn
from shiny import run_app


def run_proxy():
    uvicorn.run("src.orchestrator.proxy:app", host="0.0.0.0", port=12340, reload=False)


def run_dashboard():
    # If dashboard.py defines an ASGI app named `app`, you can also do:
    # uvicorn.run("src.dashboard.dashboard:app", host="0.0.0.0", port=12341)
    run_app("src/dashboard/dashboard.py", host="0.0.0.0", port=12341)


if __name__ == "__main__":
    t1 = threading.Thread(target=run_proxy, daemon=True)
    t2 = threading.Thread(target=run_dashboard, daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
