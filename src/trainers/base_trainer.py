import itertools

import torch
from tqdm import tqdm

from src.metrics import get_au_fa_fr


class BaseTrainer:
    def __init__(self, opt, train_loader, val_loader=None,
                 device=torch.device('cpu'), max_grad_norm=5.0):
        self.opt = opt
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_grad_norm = max_grad_norm

        self.best_score = 1.0
        self.best_state = None

    def clip_grad_norm(self):
        parameters = itertools.chain.from_iterable(
            (param_group['params'] for param_group in self.opt.param_groups)
        )
        torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)

    def eval_mode(self):
        '''
        this function is expected to switch trainer to eval mode
        '''
        raise NotImplementedError

    def save_best_state(self):
        '''
        this function is expected to update self.best_state
        '''
        raise NotImplementedError

    def train(self, n_epochs, make_plots=True):
        if make_plots:
            from IPython.display import clear_output
            import matplotlib.pyplot as plt
        val_history = []
        for n in range(1, n_epochs + 1):
            self.train_epoch()
            au_fa_fr = self.validation()
            val_history.append(au_fa_fr)
            if au_fa_fr < self.best_score:
                self.best_score = au_fa_fr
                self.save_best_state()
            if make_plots:
                clear_output()
                plt.plot(val_history)
                plt.ylabel('Metric')
                plt.xlabel('Epoch')
                plt.grid()
                plt.show()
            print(f'END OF EPOCH {n}, val metric: {au_fa_fr}')
        return val_history

    def train_mode(self):
        '''
        this function is expected to switch trainer to train mode
        '''
        raise NotImplementedError

    def train_batch(self, batch, labels):
        '''
        this function is expected to return batch loss
        '''
        raise NotImplementedError

    @torch.no_grad()
    def validate_batch(self, batch, labels):
        '''
        this function is expected to return probs, i.e.,
        probs[i] = P[label(batch[i]) == 1]
        '''
        raise NotImplementedError

    def train_epoch(self):
        self.train_mode()
        for batch, labels in tqdm(self.train_loader,
                                  total=len(self.train_loader)):
            batch, labels = batch.to(self.device), labels.to(self.device)
            loss = self.train_batch(batch, labels)
            self.opt.zero_grad()
            loss.backward()
            self.clip_grad_norm()
            self.opt.step()

    @torch.no_grad()
    def validation(self):
        self.eval_mode()
        all_probs, all_labels = [], []
        for batch, labels in tqdm(self.val_loader,
                                  total=len(self.val_loader)):
            batch, labels = batch.to(self.device), labels.to(self.device)
            probs = self.validate_batch(batch, labels)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
        return get_au_fa_fr(torch.cat(all_probs, dim=0), all_labels)
