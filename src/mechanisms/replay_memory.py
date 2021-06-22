import random
import logging

from collections import namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


logger = logging.getLogger(__name__)


class ReplayMemory:
    """
    We'll be using experience replay memory for training our DQN.
    It stores the transitions that the agent observes,
    allowing us to reuse this data later.
    By sampling from it randomly, the transitions that build up a batch are decorrelated.
    It has been shown that this greatly stabilizes and improves the DQN training procedure.

    For this, we're going to need two classses:
        * Transition - a named tuple representing a single transition in our environment
        * ReplayMemory - a cyclic buffer of bounded size that holds the transitions observed recently.
          It also implements a .sample() method for selecting a random batch of transitions for training.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0

        logger.debug(f"Initialized Replay Memory with capacity '{capacity}'")

    def push(self, *args):
        """
        Saves a transition
        transition -> (state t, action t, reward t, state t+1)
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


