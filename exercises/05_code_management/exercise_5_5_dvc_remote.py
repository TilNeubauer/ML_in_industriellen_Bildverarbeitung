"""Exercise 5.5: DVC-Remote lokal und optional für den Kurs konfigurieren.

Aufgabenstellung:
1. Synchronisiere Daten vom vorgegebenen Remote.
2. Lege im Sakai-Remote einen persönlichen students-remote-Unterordner an und
   synchronisiere Experimente mit DVC und Git.
3. Nutze ein lokales Verzeichnis als Remote und untersuche dessen Struktur.

Dieses Skript testet Teil 3 vollständig. Für Sakai setze COURSE_REMOTE auf die
persönliche URL und verwende anschließend `dvc remote add -d course ...`.
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
    base = Path("results/exercise_5_5").resolve(); root = base / "project"; remote = base / "remote"
    if not root.exists():
        root.mkdir(parents=True)
        run("git", "init", cwd=root); run("dvc", "init", cwd=root)
        (root / "data.txt").write_text("lokale DVC-Remote-Daten\n")
        run("dvc", "add", "data.txt", cwd=root); run("dvc", "remote", "add", "-d", "local", str(remote), cwd=root)
        run("dvc", "push", cwd=root)
        assert any(remote.rglob("*")), "Remote enthält keinen DVC-Cache"
        (root / "data.txt").unlink(); run("dvc", "checkout", cwd=root)
        assert (root / "data.txt").read_text() == "lokale DVC-Remote-Daten\n"
    print(f"Lokales Remote getestet; Cache-Struktur liegt unter: {remote}")


if __name__ == "__main__": main()
