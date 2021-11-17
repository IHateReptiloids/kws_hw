import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.metrics import get_au_fa_fr


def count_parameters(model):
    return sum([p.numel() for p in model.parameters()])


def train(n_epochs, model, opt, loaders, wave2specs, device, make_plots=True):
    if make_plots:
        from IPython.display import clear_output
        import matplotlib.pyplot as plt

    train_loader, val_loader = loaders
    train_wave2spec, val_wave2spec = wave2specs
    val_history = []
    best_score = 1.0
    best_model = None
    for n in range(1, n_epochs + 1):
        train_epoch(model, opt, train_loader, train_wave2spec, device)
        au_fa_fr = validation(model, val_loader, val_wave2spec, device)
        val_history.append(au_fa_fr)
        if au_fa_fr < best_score:
            best_score = au_fa_fr
            best_model = model.state_dict()
        if make_plots:
            clear_output()
            plt.plot(val_history['val_metric'])
            plt.ylabel('Metric')
            plt.xlabel('Epoch')
            plt.grid()
            plt.show()
        print('END OF EPOCH', n)
    return best_score, best_model


def train_epoch(model, opt, loader, wave2spec, device):
    model.train()
    for batch, labels in tqdm(loader, total=len(loader)):
        batch, labels = batch.to(device), labels.to(device)
        batch = wave2spec(batch)

        opt.zero_grad()

        logits = model(batch)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        opt.step()

        # logging
        probs = F.softmax(logits, dim=-1)
        argmax_probs = torch.argmax(probs, dim=-1)
        acc = torch.sum(argmax_probs == labels) / torch.numel(argmax_probs)

    return acc


@torch.no_grad()
def validation(model, loader, wave2spec, device):
    model.eval()

    all_probs, all_labels = [], []
    for batch, labels in tqdm(loader):
        batch, labels = batch.to(device), labels.to(device)
        batch = wave2spec(batch)

        output = model(batch)
        probs = F.softmax(output, dim=-1)

        # logging
        all_probs.append(probs[:, 1].cpu())
        all_labels.append(labels.cpu())

    # area under FA/FR curve for whole loader
    au_fa_fr = get_au_fa_fr(torch.cat(all_probs, dim=0).cpu(), all_labels)
    return au_fa_fr
