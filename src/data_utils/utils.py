import torch

from .collator import Collator
from .sampler import get_sampler


def get_dataloaders(train_ds, val_ds, batch_size,
                    num_workers=2, pin_memory=True):
    train_sampler = get_sampler(train_ds.csv['label'].values)

    train_loader = torch.utils.data.DataLoader(
                            train_ds, batch_size=batch_size,
                            shuffle=False, collate_fn=Collator(),
                            sampler=train_sampler,
                            num_workers=num_workers, pin_memory=pin_memory)

    val_loader = torch.utils.data.DataLoader(
                            val_ds, batch_size=batch_size,
                            shuffle=False, collate_fn=Collator(),
                            num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader
