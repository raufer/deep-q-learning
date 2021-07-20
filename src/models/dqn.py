import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import config


class DQN(nn.Module):
    """
    Assumption: the environment is deterministic
    so all equations presented here are also formulated deterministically for the sake of simplicity.

    In the reinforcement learning literature, they would also contain expectations
    over stochastic transitions in the environment.

    Our aim is to train a policy that tries to maximize the discounter, cumulative reward
    R = sum_{t=t0}^{inf} 𝛾^t * r_t

    The discount, 𝛾 , should be a constant between  0  and  1  that ensures the sum converges.
    It makes rewards from the uncertain, far future, less important for our agent
    than the ones in the near future that it can be more confident about

    The main idea behind Q-learning is:

    If we had a function Q* :: (S, A) -> R (scalar) that could tell us the real return of
    taking an action A at the state S, then we could easily construct an optimal policy:

    policy*(s) = argmax {a} Q*(S, a)
    This policy would always maximize our rewards

    However, we dont know everything about the world, so we do not have direct access to Q*
    Nevertheless, We can use function approximation techniques to approximate Q*

    For the training update rule, we'll use the fact that every function Q for some policy
    obeys the Bellman Equation:

    Q_pi(s, a) = r + gamma * max {a'} Q_pi(s', a')

    The difference between the two sides of the equality is known as the temporal
    difference error

    delta = Q(s,a) - (r + gamma max {a} Q(s', a))

    To minimize this error, we'll use the Hubber loss:
    * MSE when the error is small (< 1)
    * MAE when the error is large (> 1)
    (more robust to outliers)

    This error is calculated over a batch of transitions B
    sampled from the replay memory

    L = 1 / |B| * sum {(s, a, s', r) in B} L(delta)

    with L(delta) =
        1/2 delta**2    for |delta| < 1
        |delta| - 1/2   otherwise

    Q-network

    Our model is a convolutional neural network that takes as input
    the different between the current and previous screen patches.

    It has two outputs representing Q(s, left) and Q(s, right),
    where s is the input to the network.

    In effect, the network is trying to predict the quality/value of
    taking each action given the current input
    """

    def __init__(self, h, w, outputs):

        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32

        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

