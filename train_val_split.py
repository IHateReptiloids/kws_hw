from pathlib import Path

from src.configs import DefaultConfig
from src.data_utils import SpeechCommandsDataset
from src.utils import seed_all

DATA_DIR = Path('data/speech_commands')
TRAIN_CSV = Path('data/train.csv')
VAL_CSV = Path('data/val.csv')


ds = SpeechCommandsDataset(DATA_DIR, keywords=DefaultConfig.keyword)
seed_all()
train_ds, val_ds = ds.train_val_split(train_frac=0.8)
train_ds.csv.to_csv(TRAIN_CSV, index=False)
val_ds.csv.to_csv(VAL_CSV, index=False)
