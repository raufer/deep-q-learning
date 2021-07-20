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


def plot_loss(loss_log):
    """
    The plot will be underneath the cell containing the main training loop,
    and will update after every episode.
    """
    plt.figure(3)
    plt.clf()

    plt.title('Huber Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')

    plt.plot(loss_log)

    # pause a bit so that plots are updated
    plt.pause(0.001)


def plot_gradient(gradient_log):
    plt.figure(4)
    plt.clf()
    plt.title('Gradient average at the different layers')
    plt.xlabel('Step')
    plt.ylabel('W Magnitude')
    plt.plot(gradient_log['head'], label='head')
    plt.legend(loc="upper right")

    plt.figure(5)
    plt.clf()
    plt.title('Gradient average at the different layers')
    plt.xlabel('Step')
    plt.ylabel('W Magnitude')
    plt.plot(gradient_log['conv1'], label='conv1')
    plt.plot(gradient_log['conv2'], label='conv2')
    plt.plot(gradient_log['conv3'], label='conv3')
    plt.legend(loc="upper right")

    # pause a bit so that plots are updated
    plt.pause(0.001)


def plot_lr(lr_log):
    plt.figure(6)
    plt.clf()

    plt.title('Learning Rate Evolution')
    plt.xlabel('Step')
    plt.ylabel('lr')

    plt.plot(lr_log)

    # pause a bit so that plots are updated
    plt.pause(0.001)
