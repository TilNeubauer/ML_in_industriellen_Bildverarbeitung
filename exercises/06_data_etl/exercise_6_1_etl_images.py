"""Exercise 6.1: ETL für Cats-vs-Dogs-Bilder.

Aufgabenstellung:
1. Extrahiere Einzelbilder aus .mat-Dateien, getrennt nach Klassen.
2. Teile sie in Train/Test auf.
3. Erlaube Bildtransformationen.
4. Speichere transformierte Bilder verlustfrei.
5. Lade Klassen sowie Train/Test allein aus dem Ordner data/raw.

Aufruf: pdm run python exercises/06_data_etl/exercise_6_1_etl_images.py
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split

RESULTS = Path("results/exercise_6_1")
SEED = 6020


def extract_mat(mat_file, key, class_name, output, test_size=.25):
    """Speichert Bilder einer (H, W, N)-Matrix verlustfrei als PNG in Train/Test."""
    images = loadmat(mat_file)[key]
    train, test = train_test_split(range(images.shape[-1]), test_size=test_size, random_state=SEED)
    for split, indices in (("train", train), ("test", test)):
        folder = output / class_name / split; folder.mkdir(parents=True, exist_ok=True)
        for index in indices:
            Image.fromarray(images[..., index].astype("uint8")).save(folder / f"{class_name}{index}.png")


def transform(image):
    """Beispieltransformation: horizontal spiegeln, Eingabe kann PIL- oder NumPy-Bild sein."""
    return ImageOps.mirror(Image.fromarray(np.asarray(image)))


def load_folders(root, apply_transform=False):
    """Ermittelt Klassen und Splits aus der Ordnerstruktur automatisch."""
    data = {}
    for class_dir in root.iterdir():
        for split_dir in class_dir.iterdir():
            data[class_dir.name, split_dir.name] = [transform(Image.open(path)) if apply_transform else Image.open(path) for path in split_dir.glob("*.png")]
    return data


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for name in ("cat", "dog"):
        file = RESULTS / f"{name}.mat"; savemat(file, {f"{name}_images": rng.integers(0, 256, (16, 16, 12), dtype=np.uint8)})
        extract_mat(file, f"{name}_images", name, RESULTS / "data" / "raw")
    loaded = load_folders(RESULTS / "data" / "raw", apply_transform=True)
    print("Geladene Bilder pro Klasse/Split:", {key: len(value) for key, value in loaded.items()})
    assert sum(map(len, loaded.values())) == 24


if __name__ == "__main__": main()
