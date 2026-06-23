"""Exercise 5.4: DVC-Experimente visualisieren.

Aufgabenstellung: Verwende `dvc plots`, um Experimente und deren Unterschiede zu
visualisieren.

Aufruf: pdm run python exercises/05_code_management/exercise_5_4_dvc_plots.py
"""

import subprocess
import os
from pathlib import Path


def run(*command, cwd):
    environment = os.environ | {"DVC_GLOBAL_CONFIG_DIR": str(Path("results/.dvc_global").resolve())}
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, env=environment)
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)}\n{result.stderr}")
    return result


def main():
    root = Path("results/exercise_5_4/dvc_plots_demo").resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not (root / ".dvc").exists():
        run("git", "init", cwd=root); run("dvc", "init", cwd=root)
        (root / "metrics.csv").write_text("epoch,accuracy\n1,0.70\n2,0.82\n3,0.89\n")
        run("dvc", "add", "metrics.csv", cwd=root)
        output = root / "plots"; run("dvc", "plots", "show", "metrics.csv", "-o", str(output), cwd=root)
        assert (output / "index.html").exists()
        print(f"DVC-Plot erzeugt: {output / 'index.html'}")


if __name__ == "__main__": main()
