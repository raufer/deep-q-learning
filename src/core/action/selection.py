import torch
import math
import random

from src import device
from src.config import config


def e_greedy_selection(state, step_i, policy_net):
    """
    We'll select an action accordingly to an epsilon greedy policy.
    Simply put, we'll sometimes use our model for choosing the action,
    and sometimes we'll just sample one uniformly.

    The probability of choosing a random action will start at EPS_START
    and will decay exponentially towards EPS_END. EPS_DECAY controls the rate of the decay.
    """
    eps_threshold = config.EPS_END + (config.EPS_START - config.EPS_END) * math.exp(-1.0 * step_i / config.EPS_DECAY)

    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        choice = random.randrange(2)
        return torch.tensor([[choice]], device=device, dtype=torch.long)

