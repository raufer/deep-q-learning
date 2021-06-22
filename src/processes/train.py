import os
import torch
import logging
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src import device
from src.config import config
from src.core.action.selection import e_greedy_selection
from src.models.dqn import DQN
from src.mechanisms.replay_memory import Transition
from src.mechanisms.replay_memory import ReplayMemory

from itertools import count

from src.ops.plot import plot_durations
from src.ops.screen import get_screen
from src.utils.checkpoints import save_checkpoint
from src.utils.directories import make_run_dir

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

    # transpose the batch
    batch = Transition(*zip(*transitions))

    # compute a mask of non-final states and concatenate the batch elements
    # true = non-final-state
    non_final_mask = tuple(map(lambda s: s is not None, batch.next_state))
    non_final_mask = torch.tensor(non_final_mask, device=device, dtype=torch.uint8)

    non_final_next_states = [s for s in batch.next_state if s is not None]
    non_final_next_states = torch.cat(non_final_next_states)

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # compute Q(s_t, a) - the model computes Q(s_t), then
    # we select columns of actions to take
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # compute V(s_{t+1}) for all next states
    next_state_values = torch.zeros(config.BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # compute the expected Q values
    # (expected return following the Q-learning algorithm)
    expected_state_action_values = (next_state_values * config.GAMMA) + reward_batch

    # compute huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def run_episode(env, memory, policy_net, target_net, optimizer, step):
    """
    Runs a single episode
    when the episode ends (our model fails), we restart the loop
    """

    # initialize the environment and state
    env.reset()

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
        optimize_model(memory, policy_net, target_net, optimizer)
        step += 1

        if done:
            break

    return step


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

    logger.info(f"Creating policy network")
    policy_net = DQN().to(device)

    logger.info(f"Creating target network")
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(capacity=10000)

    step = 0
    episode_durations = []
    num_episodes = config.NUM_EPISODES
    logger.info(f"Running '{num_episodes}' episodes")

    for i_episode in range(num_episodes):

        logger.info(f"Running episode '{i_episode}'")
        step_ = run_episode(env, memory, policy_net, target_net, optimizer, step)

        duration = step_ - step
        step = step_
        logger.info(f"Episode '{i_episode}' finished in '{duration}' steps")

        episode_durations.append(duration + 1)

        if i_episode % config.TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        plot_durations(episode_durations)

    logger.info(f"Training completed")
    env.render()
    env.close()

    save_checkpoint(os.path.join(output_path, 'model'), policy_net)


if __name__ == '__main__':

    import gym

    env = gym.make('CartPole-v0').unwrapped

    output_dir = '/Users/raulferreira/rl/dqn/output/model'

    training_job(env)
