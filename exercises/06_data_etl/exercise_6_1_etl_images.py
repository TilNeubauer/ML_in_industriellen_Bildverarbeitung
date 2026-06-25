"""Exercise 6.1: Extension of the ETL.

From the original .mat files extract the single images and store them in a
lossless image format, separated by class and train/test split.
Extend transformation, storage, and loading so that data/raw is enough to
recover classes and splits automatically.

Aufruf: pdm run python exercises/06_data_etl/exercise_6_1_etl_images.py
"""

from pathlib import Path
from shutil import copyfile
from shutil import rmtree
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageOps
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

RESULTS = Path("results/exercise_6_1")
MAT_DIR = RESULTS / "mat"
RAW_DIR = RESULTS / "data" / "raw"
TRANSFORMED_DIR = RESULTS / "data" / "transformed"
CACHE_DIR = Path("results/exercise_2_8/dogs_cats_data")
BASE_URL = "https://github.com/dynamicslab/databook_python/raw/refs/heads/master/DATA"
SEED = 6020


def ensure_mat_file(animal):
    target = MAT_DIR / f"{animal}Data.mat"
    cached = CACHE_DIR / target.name

    if target.exists():
        return target

    MAT_DIR.mkdir(parents=True, exist_ok=True)
    if cached.exists():
        copyfile(cached, target)
        return target

    with urlopen(f"{BASE_URL}/{target.name}", timeout=60) as response:
        target.write_bytes(response.read())

    return target


def load_mat_images(animal):
    matrix = loadmat(ensure_mat_file(animal))[animal]
    return matrix.T.reshape(-1, 64, 64).astype(np.uint8)


def save_images(images, animal, output, test_size=0.25):
    indices = np.arange(len(images))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=SEED,
    )

    for split, split_indices in (("train", train_indices), ("test", test_indices)):
        folder = output / animal / split
        folder.mkdir(parents=True, exist_ok=True)

        for index in split_indices:
            image = Image.fromarray(images[index])
            image.save(folder / f"{animal}{index}.png")


def transform_image(image):
    image = Image.fromarray(np.asarray(image))
    return ImageOps.mirror(image)


def save_transformed_images(raw_root, output):
    for image_path in raw_root.glob("*/*/*.png"):
        relative_path = image_path.relative_to(raw_root)
        target_path = output / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        transformed = transform_image(Image.open(image_path))
        transformed.save(target_path)


def load_image_folders(root):
    data = {}

    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for split_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            images = [
                Image.open(path).copy()
                for path in sorted(split_dir.glob("*.png"))
            ]
            data[class_dir.name, split_dir.name] = images

    return data


def main():
    for folder in (RAW_DIR, TRANSFORMED_DIR):
        if folder.exists():
            rmtree(folder)

    for animal in ("cat", "dog"):
        images = load_mat_images(animal)
        save_images(images, animal, RAW_DIR)

    save_transformed_images(RAW_DIR, TRANSFORMED_DIR)
    loaded = load_image_folders(RAW_DIR)

    counts = {key: len(images) for key, images in loaded.items()}
    print("Geladene Bilder pro Klasse/Split:", counts)

    assert counts == {
        ("cat", "test"): 20,
        ("cat", "train"): 60,
        ("dog", "test"): 20,
        ("dog", "train"): 60,
    }


if __name__ == "__main__":
    main()
