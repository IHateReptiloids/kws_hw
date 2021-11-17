from copy import deepcopy

import torch
import torch.nn.functional as F

from .base_trainer import BaseTrainer


class DefaultTrainer(BaseTrainer):
    def __init__(
        self,
        *,
        n_epochs,
        opt,
        train_loader,
        val_loader,
        device,
        model,
        train_wave2spec,
        val_wave2spec,
    ):
        self.model = model.to(device)
        self.train_wave2spec = train_wave2spec
        self.val_wave2spec = val_wave2spec
        super().__init__(
            n_epochs,
            opt,
            train_loader,
            val_loader,
            device,
        )

    def eval_mode(self):
        self.model = self.model.eval()

    def save_best_state(self):
        self.best_state = deepcopy(self.model.state_dict())

    def train_mode(self):
        self.model = self.model.train()

    def train_batch(self, batch, labels):
        batch = self.train_wave2spec(batch)
        return F.cross_entropy(self.model(batch), labels)

    @torch.no_grad()
    def validate_batch(self, batch, labels):
        batch = self.val_wave2spec(batch)
        probs = F.softmax(self.model(batch), dim=-1)[:, 1]
        return probs
