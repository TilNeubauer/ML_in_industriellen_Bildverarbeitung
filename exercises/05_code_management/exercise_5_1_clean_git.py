"""Exercise 5.1: 
    Implement a function (or a decorator) in Python that uses GitPython 
    (alternatives or the plain shell can also be used) to introduce a safeguard such that 
    training can only be called if the repository is not dirty.
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
