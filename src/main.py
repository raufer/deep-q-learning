import gym
import math
import random
import numpy as np
import matplotlib

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from src.ops.screen import get_screen
from collections import namedtuple
from itertools import count
from PIL import Image

matplotlib.interactive(True)

env = gym.make('CartPole-v0').unwrapped


if __name__ == '__main__':

    import os

    images_dir = '/Users/raulferreira/rl/dqn/output/images'

    env.reset()

    plt.figure()
    img = get_screen(env).cpu().squeeze(0).permute(1, 2, 0).numpy()
    plt.imshow(img, cmap='gray')
    plt.show(block=True)
    # plt.imsave(os.path.join(images_dir, 'sample-screen.png'), img)
    # plt.savefig(os.path.join(images_dir, 'sample-screen.png'))

