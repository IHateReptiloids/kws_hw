from pathlib import Path

import torch
from torch import distributions
import torchaudio

DEFAULT_NOISES = [
    'white_noise.wav',
    'dude_miaowing.wav',
    'doing_the_dishes.wav',
    'exercise_bike.wav',
    'pink_noise.wav',
    'running_tap.wav'
]


class DefaultWaveAugmentations:
    def __init__(self, data_dir: str):
        data_dir = Path(data_dir)
        self.noises = [
            torchaudio.load(str(data_dir / '_background_noise_' / noise))[0]
            .squeeze()
            for noise in DEFAULT_NOISES
        ]

    def add_rand_noise(self, audio):
        # randomly choose noise
        noise_num = torch.randint(low=0, high=len(
            self.noises), size=(1,)).item()
        noise = self.noises[noise_num]

        noise_level = torch.Tensor([1])  # [0, 40]

        noise_energy = torch.norm(noise)
        audio_energy = torch.norm(audio)
        alpha = (audio_energy / noise_energy) * \
            torch.pow(10, -noise_level / 20)

        start = torch.randint(
            low=0,
            high=max(int(noise.size(0) - audio.size(0) - 1), 1),
            size=(1,)
        ).item()
        noise_sample = noise[start: start + audio.size(0)]

        audio_new = audio + alpha * noise_sample
        audio_new.clamp_(-1, 1)
        return audio_new

    def __call__(self, wav):
        aug_num = torch.randint(low=0, high=4, size=(1,)).item()
        augs = [
            lambda x: x,
            lambda x: (x + distributions.Normal(0, 0.01).sample(x.size()))
            .clamp_(-1, 1),
            lambda x: torchaudio.transforms.Vol(.25)(x),
            lambda x: self.add_rand_noise(x)
        ]

        return augs[aug_num](wav)
