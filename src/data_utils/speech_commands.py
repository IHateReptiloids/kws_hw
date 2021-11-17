from pathlib import Path
import shutil
from typing import Callable, List, Optional, Union

import pandas as pd
from torch.utils.data import Dataset
import torchaudio
import wget

URL = 'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz'


class SpeechCommandsDataset(Dataset):
    def __init__(
        self,
        transform: Optional[Callable] = None,
        path2dir: str = None,
        keywords: Union[str, List[str]] = None,
        csv: Optional[pd.DataFrame] = None
    ):
        if path2dir is None and csv is None:
            raise ValueError('You must specify either path2dir or csv')

        self.transform = transform

        if path2dir is not None:
            path2dir = Path(path2dir)
            if not path2dir.exists() or len(list(path2dir.iterdir())) == 0:
                self._download(path2dir)

        if csv is None:
            keywords = keywords if isinstance(keywords, list) else [keywords]

            all_keywords = [
                p.stem for p in path2dir.glob('*')
                if p.is_dir() and not p.stem.startswith('_')
            ]

            triplets = []
            for keyword in all_keywords:
                paths = (path2dir / keyword).rglob('*.wav')
                if keyword in keywords:
                    for path2wav in paths:
                        triplets.append((path2wav.as_posix(), keyword, 1))
                else:
                    for path2wav in paths:
                        triplets.append((path2wav.as_posix(), keyword, 0))

            self.csv = pd.DataFrame(
                triplets,
                columns=['path', 'keyword', 'label']
            )
        else:
            self.csv = csv

    def __getitem__(self, index: int):
        instance = self.csv.iloc[index]

        path2wav = instance['path']
        wav, sr = torchaudio.load(path2wav)
        wav = wav.sum(dim=0)

        if self.transform:
            wav = self.transform(wav)

        return {
            'wav': wav,
            'keywors': instance['keyword'],
            'label': instance['label']
        }

    def __len__(self):
        return len(self.csv)

    def _download(self, path2dir: Path):
        path2dir.mkdir(parents=True, exist_ok=True)
        archive_path = path2dir / 'archive.tar.gz'
        if archive_path.exists():
            raise RuntimeError('archive.tar.gz exists in root directory')
        wget.download(URL, str(archive_path))
        shutil.unpack_archive(str(archive_path), str(path2dir))
        archive_path.unlink()
