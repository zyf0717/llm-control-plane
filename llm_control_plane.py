import multiprocessing

import uvicorn
from shiny import run_app


def run_proxy():
    uvicorn.run("proxy:app", host="0.0.0.0", port=12340, reload=False)


def run_dashboard():
    run_app("dashboard.py", host="0.0.0.0", port=12341)


if __name__ == "__main__":
    p1 = multiprocessing.Process(target=run_proxy)
    p2 = multiprocessing.Process(target=run_dashboard)
    p1.start()
    p2.start()
    p1.join()
    p2.join()
