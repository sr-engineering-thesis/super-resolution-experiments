import pathlib

DATA_DIR_PATH = "data"
CHECKPOINTS_DIR_PATH = "checkpoints"
RESULTS_DIR_PATH = "results"

TRAIN_HR_DATA_PATH = pathlib.Path(DATA_DIR_PATH) / "train/hr_native"
TRAIN_LR_DATA_PATH = pathlib.Path(DATA_DIR_PATH) / "train/lr_downscaled"

TEST_HR_DATA_PATH = pathlib.Path(DATA_DIR_PATH) / "test/hr_native"
TEST_LR_DATA_PATH = pathlib.Path(DATA_DIR_PATH) / "test/lr_downscaled"