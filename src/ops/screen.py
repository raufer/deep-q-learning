import torch
import numpy as np
import torchvision.transforms as T

from PIL import Image

from src import device
from src.constants import SCREEN_WIDTH


resize_op = T.Compose([
    T.ToPILImage(),
    T.Resize(40, interpolation=Image.CUBIC),
    T.ToTensor()
])


def get_cart_location(env) -> int:
    world_width = env.x_threshold * 2
    scale = SCREEN_WIDTH / world_width
    # middle of the cart
    return int(env.state[0] * scale + SCREEN_WIDTH / 2.0)


def get_screen(env):

    # transpose into torch order (CHW)
    screen = env.render(mode='rgb_array').transpose( (2, 0, 1))

    # Strip off the top and bottom of the screen
    screen = screen[:, 160:320]
    view_width = 320

    cart_location = get_cart_location(env)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (SCREEN_WIDTH - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)

    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]

    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255

    screen = torch.from_numpy(screen)

    # Resize, and add a batch dimension (BCHW)
    return resize_op(screen).unsqueeze(0).to(device)




