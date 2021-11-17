from pathlib import Path
import shutil

import pandas as pd
import torch
import torchaudio
import wget

URL = 'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz'


class SpeechCommandsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path2dir,
        transform=None,
        keywords=None,
        csv=None
    ):
        if keywords is None and csv is None:
            raise ValueError('You must specify either keywords or csv')

        self.transform = transform

        path2dir = Path(path2dir)
        self.path2dir = path2dir
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
                        triplets.append((str(path2wav.relative_to(path2dir)),
                                         keyword, 1))
                else:
                    for path2wav in paths:
                        triplets.append((str(path2wav.relative_to(path2dir)),
                                         keyword, 0))

            self.csv = pd.DataFrame(
                triplets,
                columns=['path', 'keyword', 'label']
            )
        else:
            self.csv = csv

        self.csv.sort_values(by='path', inplace=True, ignore_index=True)

    def __getitem__(self, index: int):
        instance = self.csv.iloc[index]

        path2wav = self.path2dir / instance['path']
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

    def train_val_split(self, train_frac, augs=None):
        indices = torch.randperm(len(self))
        train_sz = int(len(self) * train_frac)
        train_indices = indices[:train_sz]
        val_indices = indices[train_sz:]

        train_df = self.csv.iloc[train_indices].reset_index(drop=True)
        val_df = self.csv.iloc[val_indices].reset_index(drop=True)
        train_set = SpeechCommandsDataset(self.path2dir, csv=train_df,
                                          transform=augs)
        val_set = SpeechCommandsDataset(self.path2dir, csv=val_df)
        return train_set, val_set

    def _download(self, path2dir: Path):
        path2dir.mkdir(parents=True, exist_ok=True)
        archive_path = path2dir / 'archive.tar.gz'
        if archive_path.exists():
            raise RuntimeError('archive.tar.gz exists in root directory')
        wget.download(URL, str(archive_path))
        shutil.unpack_archive(str(archive_path), str(path2dir))
        archive_path.unlink()
