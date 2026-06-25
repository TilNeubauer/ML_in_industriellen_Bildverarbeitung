"""Exercise 5.3 (dvc queue)

Checkout the option --queue for dvc exp run. Use it to plan various
experiments with different parameters from the params.yaml file.

Run them via dvc queue start.

Aufruf: pdm run python exercises/05_code_management/exercise_5_3_dvc_queue.py
"""

import os
import subprocess
import time
from pathlib import Path

BASE_ROOT = Path("results/exercise_5_3/dvc_queue_demo").resolve()
ROOT = BASE_ROOT
DVC_ENV = os.environ | {
    "DVC_GLOBAL_CONFIG_DIR": str(Path("results/.dvc_global").resolve()),
    "DVC_SITE_CACHE_DIR": str(Path("results/.dvc_site_cache").resolve()),
    "PYTHONIOENCODING": "utf-8",
}


def choose_root():
    if not BASE_ROOT.exists():
        return BASE_ROOT

    for index in range(1, 100):
        candidate = BASE_ROOT.with_name(f"{BASE_ROOT.name}_{index}")
        if not candidate.exists():
            return candidate

    raise RuntimeError("Kein freier Demo-Ordner gefunden.")


def run(*command):
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env=DVC_ENV, timeout=45)
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)}\n{result.stderr}")
    return result.stdout


def write_demo_files():
    (ROOT / "params.yaml").write_text("train:\n  C: 1\n", encoding="utf-8")
    (ROOT / "train.py").write_text(
        "import json, yaml\n"
        "params = yaml.safe_load(open('params.yaml', encoding='utf-8'))\n"
        "json.dump({'C': params['train']['C']}, open('metrics.json', 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    (ROOT / "dvc.yaml").write_text(
        "stages:\n"
        "  train:\n"
        "    cmd: python train.py\n"
        "    params: [train.C]\n"
        "    metrics: [metrics.json]\n",
        encoding="utf-8",
    )


def create_demo_repo():
    global ROOT
    ROOT = choose_root()
    print(f"Demo-Repo: {ROOT}", flush=True)

    ROOT.mkdir(parents=True)
    print("Initialisiere Git und DVC ...", flush=True)
    run("git", "init")
    run("git", "config", "user.email", "student@example.com")
    run("git", "config", "user.name", "Student")
    run("dvc", "init")

    write_demo_files()
    run("git", "add", ".")
    run("git", "commit", "-m", "baseline")


def wait_for_queue():
    time.sleep(20)


def main():
    create_demo_repo()

    print("Plane Experimente mit dvc exp run --queue ...", flush=True)
    for value in (0.1, 1, 10):
        run("dvc", "exp", "run", "--queue", "-S", f"train.C={value}")

    print("Starte DVC-Queue ...", flush=True)
    run("dvc", "queue", "start", "-j", "1")
    wait_for_queue()
    run("dvc", "queue", "stop")

    experiments = run("dvc", "exp", "show", "--no-pager")
    assert "0.1" in experiments and "10" in experiments
    print(f"DVC-Queue-Experimente wurden ausgefuehrt: {ROOT}")


if __name__ == "__main__":
    main()
