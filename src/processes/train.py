import os
import torch
import logging
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import matplotlib.pyplot as plt

from src import device
from src.config import config
from src.core.action.selection import e_greedy_selection
from src.models.dqn import DQN
from src.mechanisms.replay_memory import Transition
from src.mechanisms.replay_memory import ReplayMemory

from src.ops.plot import plot_durations
from src.ops.plot import plot_loss
from src.ops.plot import plot_gradient
from src.ops.plot import plot_lr

from src.ops.screen import get_screen
from src.utils.checkpoints import save_checkpoint
from src.utils.directories import make_run_dir

from itertools import count
from collections import deque


logger = logging.getLogger(__name__)


def optimize_model(memory: ReplayMemory, policy_net, target_net, optimizer):
    """
    Performs a single step of optimization.

    1. First we sample a batch
    2. Concatenates all of the tensors into a single one
    3. Computes Q(s_t, a_t) and V(s_{t+1}) = max {a} Q(S_{t+1}, a) and combines them into our loss

    We set V(s) = 0 if s is a terminal state

    For added stability we use the target network to compute V(s_{t+1})
    The target network has its weights frozen most of the time, but
    is updated with the policy network's weights every so often
    (This is usually a set numbers of steps but we shall use episodes for simplicity)

    """

    # still not enough examples in the memory to process a batch
    if len(memory) < config.BATCH_SIZE:
            return None

    transitions = memory.sample(config.BATCH_SIZE)

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(config.BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * config.GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss


def run_episode(env, memory, policy_net, target_net, optimizer, step, update_target=True):
    """
    Runs a single episode
    when the episode ends (our model fails), we restart the loop
    """

    # initialize the environment and state
    env.reset()

    loss_log = []

    last_screen = get_screen(env)
    current_screen = get_screen(env)

    state = current_screen - last_screen

    for t in count():

        # select and execute an action
        action = e_greedy_selection(state, step, policy_net)

        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # observe new state
        last_screen = current_screen
        current_screen = get_screen(env)
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # store the transition in memory
        memory.push(state, action, next_state, reward)

        # move to the next state
        state = next_state

        # perform one step of the optimisation
        loss = optimize_model(memory, policy_net, target_net, optimizer)

        step += 1
        if loss:
            loss_log.append(loss.item())

        if update_target:
            if step < 5000:
                if step % 500 == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            else:
                if step % 500 == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        if done:
            break

    return step, loss_log


def training_job(env):
    """
    Main training loop

    A separate target network, Q^ provides update targets to the main
    network, decoupling the feedback resulting from the network
    generating its own targets.

    Q^ is identical to the main network except its parameters W' are
    updated to match W every X iterations

    This is required because if the same network is generating
    the next target Q-values, such updates can oscillate or diverge

    1. at the beginning we reset the environment and initialize the state Tensor
    2. then, we sample an action
    3. execute the action
    4. observe the next screen and the reward (always 1, i.e. keep the pole from failing)
    5. optimize the model once
    6. when the episode ends (our model fails), we restart the loop
    """
    logger.info(f"Process: 'training'")

    output_path = make_run_dir(output_dir)

    env.reset()
    init_screen = get_screen(env)
    _, _, height, width = init_screen.shape
    logger.info(f"Screen size '{height} x {width}'")

    n_actions = env.action_space.n
    logger.info(f"Cardinality of the action space '{n_actions}'")

    logger.info(f"Creating policy network")
    policy_net = DQN(height, width, n_actions).to(device)

    logger.info(f"Creating target network")
    target_net = DQN(height, width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(capacity=config.MEMORY_SIZE)

    step = 0
    avg_loss = None
    update_target = True
    episode_durations = []

    loss_log = []

    gradient_log = {
        'conv1': [],
        'conv2': [],
        'conv3': [],
        'head': []
    }

    lr_log = []

    num_episodes = config.NUM_EPISODES
    logger.info(f"Running '{num_episodes}' episodes")

    for i_episode in range(num_episodes):

        logger.info(f"Running episode '{i_episode}'")
        step_, batch_loss_log = run_episode(env, memory, policy_net, target_net, optimizer, step, update_target)

        loss_log.extend(batch_loss_log)

        duration = step_ - step
        step = step_
        logger.info(f"Episode '{i_episode}' finished in '{duration}' steps")

        if len(episode_durations) > 100:
            avg_last = int(sum(episode_durations[-100:]) / 100)
            logger.info(f"Average episode length of the last 100 episodes '{avg_last}'")

            if avg_last >= 125:
                update_target=False

        episode_durations.append(duration + 1)

        gradient_flowing = policy_net.conv1.weight.grad is not None
        if gradient_flowing:
            conv1_grad = policy_net.conv1.weight.grad.view(-1).mean()
            conv2_grad = policy_net.conv2.weight.grad.view(-1).mean()
            conv3_grad = policy_net.conv3.weight.grad.view(-1).mean()
            head_grad = policy_net.head.weight.grad.view(-1).mean()

            gradient_log['conv1'].append(conv1_grad)
            gradient_log['conv2'].append(conv2_grad)
            gradient_log['conv3'].append(conv3_grad)
            gradient_log['head'].append(head_grad)

        avg_lr = [v['square_avg'].view(-1).mean().item() for k, v in optimizer.state_dict()['state'].items()]
        if len(avg_lr):
            avg_lr = sum(avg_lr)/len(avg_lr)
        else:
            avg_lr = 0
        lr_log.append(avg_lr)

        # plots
        plot_durations(episode_durations)
        plot_loss(loss_log)
        plot_gradient(gradient_log)
        plot_lr(lr_log)

    logger.info(f"Training completed")
    env.render()
    env.close()

    save_checkpoint(os.path.join(output_path, 'model'), policy_net)


if __name__ == '__main__':

    import gym

    env = gym.make('CartPole-v0').unwrapped

    output_dir = '/Users/raulferreira/rl/dqn/output/model'

    training_job(env)
