import os
from pathlib import Path

if os.path.exists('/kaggle/'):
    ROOT_DIR = Path('/kaggle/')
else:
    ROOT_DIR = Path(__file__).root

INPUT_DIR = ROOT_DIR / "input"
WORKING_DIR = ROOT_DIR / "working"
DATASET_DIR = INPUT_DIR / "stanford-cars-dataset"
ANNOTATIONS_DIR = INPUT_DIR / "stanford-cars-dataset-annotations"

TRAIN_DIR = WORKING_DIR / "train"
VAL_DIR = WORKING_DIR / "val"