import random

import torch


def count_parameters(model):
    return sum([p.numel() for p in model.parameters()])


def get_size_in_megabytes(model):
    num_bytes = sum(
        [p.numel() * p.element_size() for p in model.parameters()]
    )
    return num_bytes / (2 ** 20)


def seed_all(seed=3407):
    _ = torch.manual_seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    return seed
