import torch
import matplotlib.pyplot as plt


def plot_durations(episode_durations):
    """
    A helper for plotting the durations of episodes,
    along with an average over the last 100 episodes
    (the measure used in the official evaluations).

    The plot will be underneath the cell containing the main training loop,
    and will update after every episode.
    """
    plt.figure(2)
    plt.clf()

    durations_t = torch.tensor(episode_durations, dtype=torch.float)

    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Duration')

    plt.plot(durations_t.numpy())

    # take 100 episodes averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # pause a bit so that plots are updated
    plt.pause(0.001)



