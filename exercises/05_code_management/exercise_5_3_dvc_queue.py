"""Exercise 5.3: DVC-Experimente in die Queue stellen und ausführen.

Aufgabenstellung: Plane mehrere Parameterexperimente mit `dvc exp run --queue`
und starte sie mit `dvc queue start`.

Aufruf: pdm run python exercises/05_code_management/exercise_5_3_dvc_queue.py
"""

import subprocess
import time
import os
from pathlib import Path


def run(*command, cwd):
    environment = os.environ | {"DVC_GLOBAL_CONFIG_DIR": str(Path("results/.dvc_global").resolve())}
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, env=environment)
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)}\n{result.stderr}")
    return result


def main():
    root = Path("results/exercise_5_3/dvc_queue_demo").resolve()
    if root.exists():
        import shutil
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    try:
        run("git", "init", cwd=root); run("git", "config", "user.email", "student@example.com", cwd=root); run("git", "config", "user.name", "Student", cwd=root)
        run("dvc", "init", cwd=root)
        (root / "params.yaml").write_text("train:\n  C: 1\n")
        (root / "train.py").write_text("import yaml\nfrom pathlib import Path\nC=yaml.safe_load(open('params.yaml'))['train']['C']\nPath('metrics.json').write_text('{\\\"C\\\": '+str(C)+'}')\n")
        (root / "dvc.yaml").write_text("stages:\n  train:\n    cmd: python train.py\n    params: [train.C]\n    metrics: [metrics.json]\n")
        run("git", "add", ".", cwd=root); run("git", "commit", "-m", "baseline", cwd=root)
        for value in (0.1, 10): run("dvc", "exp", "run", "--queue", "-S", f"train.C={value}", cwd=root)
        run("dvc", "queue", "start", cwd=root)
        for _ in range(30):
            if "Queued" not in run("dvc", "queue", "status", cwd=root).stdout: break
            time.sleep(1)
        run("dvc", "queue", "stop", cwd=root)
        result = run("dvc", "exp", "show", "--no-pager", cwd=root).stdout
        assert "0.1" in result and "10" in result
        print(f"Zwei Parameterexperimente wurden über die DVC-Queue ausgeführt: {root}")
    finally:
        if (root / ".dvc").exists():
            run("dvc", "queue", "stop", cwd=root)


if __name__ == "__main__": main()
