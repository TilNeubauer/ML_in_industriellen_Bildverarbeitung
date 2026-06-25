"""Exercise 5.4 (dvc visualization)

dvc can help us visualize our experiments. Have a look at dvc plots to
visualize the experiments and there differences.

Unterstützt von chat-GPT

Aufruf: pdm run python exercises/05_code_management/exercise_5_4_dvc_plots.py


Ergebnisse: 
    können nicht im Git repo enthalten sein da sonst ein git-repo im git-repo enthalten ist. 
    Durch aufrufen der Datei wird jedoch alles automatisch erzeugt
"""

import os
import subprocess
from pathlib import Path

BASE_ROOT = Path("results/exercise_5_4/dvc_plots_demo").resolve()
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
    # Inspieriert durch: https://doc.dvc.org/user-guide/project-structure/dvcyaml-files
    # Unterstützt von Chat-GPT


    (ROOT / "params.yaml").write_text("train:\n  learning_rate: 0.1\n", encoding="utf-8")
    (ROOT / "train.py").write_text(
        "import csv, yaml\n"
        "lr = yaml.safe_load(open('params.yaml', encoding='utf-8'))['train']['learning_rate']\n"
        "with open('metrics.csv', 'w', newline='', encoding='utf-8') as file:\n"
        "    writer = csv.writer(file)\n"
        "    writer.writerow(['epoch', 'accuracy'])\n"
        "    for epoch in range(1, 6):\n"
        "        writer.writerow([epoch, round(0.65 + lr * epoch, 3)])\n",
        encoding="utf-8",
    )
    (ROOT / "dvc.yaml").write_text(
        "stages:\n"
        "  train:\n"
        "    cmd: python train.py\n"
        "    params: [train.learning_rate]\n"
        "    plots:\n"
        "      - metrics.csv:\n"
        "          x: epoch\n"
        "          y: accuracy\n",
        encoding="utf-8",
    )


def create_demo_repo():
    global ROOT
    ROOT = choose_root()
    ROOT.mkdir(parents=True)

    run("git", "init")
    run("git", "config", "user.email", "student@example.com")
    run("git", "config", "user.name", "Student")
    run("dvc", "init")

    write_demo_files()
    run("git", "add", ".")
    run("git", "commit", "-m", "baseline")
    run("dvc", "exp", "run")


def main():
    create_demo_repo()

    for learning_rate in (0.05, 0.10, 0.15):
        run("dvc", "exp", "run", "-S", f"train.learning_rate={learning_rate}")

    output = ROOT / "plots"
    run("dvc", "plots", "diff", "--targets", "metrics.csv", "-o", str(output))

    plot = output / "index.html"
    assert plot.exists()
    print(f"DVC-Experimentvergleich erzeugt: {plot}")


if __name__ == "__main__":
    main()
