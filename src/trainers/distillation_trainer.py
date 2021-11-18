import torch
import torch.nn.functional as F

from .default_trainer import DefaultTrainer


class DistillationTrainer(DefaultTrainer):
    def __init__(
        self,
        *,
        opt,
        train_loader,
        val_loader,
        device,
        teacher,
        student,
        train_wave2spec,
        val_wave2spec,
        temp,
        alpha,
    ):
        '''
        alpha is a coef by which a hard loss is multiplied
        '''
        self.teacher = teacher.eval().to(device)
        self.temp = temp
        self.alpha = alpha
        super().__init__(
            opt=opt,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            model=student,
            train_wave2spec=train_wave2spec,
            val_wave2spec=val_wave2spec
        )

    def train_batch(self, batch, labels):
        batch = self.train_wave2spec(batch)
        with torch.no_grad():
            soft_labels = F.softmax(self.teacher(batch) / self.temp, dim=-1)
        logits = self.model(batch) / self.temp
        hard_loss = F.cross_entropy(logits, labels)
        soft_loss = F.cross_entropy(logits, soft_labels)
        return soft_loss + self.alpha * hard_loss
