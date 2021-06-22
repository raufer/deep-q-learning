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


