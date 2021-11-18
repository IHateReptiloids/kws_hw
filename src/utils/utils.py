import random

import torch


def count_parameters(model):
    return sum([p.numel() for p in model.parameters()])


def seed_all(seed=3407):
    _ = torch.manual_seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    return seed
