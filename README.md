# Deep Q Learning

Task: `CartPole-v0`


## Task Characterization

* Actions `A = {left, right}`

Environment Dynamics:

    1. The agent observes the current state of the environemnt;
    2. The agent chooses an action;
    3. The environment transitions to a new state and returns a `reward` indicating (a proxy of) the consequences of the action;
    4. If the pole falls over too far the environment terminates the episode;

As input the agent is able to sense the screen pixels at each time step. This makes the problem more difficult than one on which the agent has access to key state variables like the
position or velocity of the pole. The agent must learn these underlying unobservable states using just the screen.

Stricly speaking, we will present the state as the difference between the current screen and the previous one. This would allow the agent to take the velocity of the pole
into account using just a signle image. Otherwise we would need to stack multiple frames at the input, since not doing so would result in a task that completely violates the markov property.



## Training

In order to stabilize the training, we use an experience replay buffer which randomly samples previous transitions, and thereby
smooths the training distribution over many past behaviors. This is needed in order to alleviate the problems of correlated data and non-stationary
distributions.

Moreover, since the same network is generating the next state target Q-values that are used in updating its current Q-values, such updates can oscillate or diverge.
To contradict this , a separate, target network Qˆ provides update targets to the main network, decoupling the feedback resulting from the network generating its own target.

We parameterize the number of frames to use. Obviously the more frames, the more the state is detailed.