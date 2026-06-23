"""Exercise 5.1: Schutz vor Training in einem dirty Git-Repository.

Aufgabenstellung: Implementiere mit GitPython eine Funktion oder einen Decorator,
der Training nur in einem nicht-dirty Repository erlaubt. Optional: Prüfe, ob der
lokale Stand hinter dem Remote zurückliegt.
"""

from functools import wraps
from pathlib import Path

from git import Repo


def require_clean_repository(function):
    """Bricht vor dem Training ab, falls uncommittete Änderungen vorhanden sind."""
    @wraps(function)
    def checked(*args, repository=".", **kwargs):
        repo = Repo(Path(repository), search_parent_directories=True)
        if repo.is_dirty(untracked_files=True):
            raise RuntimeError("Training abgebrochen: Git-Repository ist dirty.")
        return function(*args, **kwargs)
    return checked


@require_clean_repository
def train_model():
    print("Repository ist sauber – Training darf starten.")


if __name__ == "__main__":
    train_model()
