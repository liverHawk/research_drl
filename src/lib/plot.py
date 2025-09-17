import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def moving_average(data, window_size):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def _plot_loss(data, path, **params):
    fig, ax = plt.subplots()
    ax.set_title("Result" if params.get("show_result") else "Training..." + f" {len(data)}")
    means = moving_average(data, 200)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.plot(means, color="lightgray", alpha=0.5)
    ax.grid()
    return fig, ax


def _plot_accuracy(data, **params):
    fig, ax = plt.subplots()
    ax.set_title("Result" if params.get("show_result") else "Training..." + f" {len(data)}")
    means = moving_average(data, 200)
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.plot(means, color="lightgray", alpha=0.5)
    ax.grid()
    return fig, ax


def _plot_weighted_f1(data, **params):
    fig, ax = plt.subplots()
    ax.set_title("Result" if params.get("show_result") else "Training..." + f" {len(data)}")
    means = moving_average(data, 200)
    ax.set_xlabel("Step")
    ax.set_ylabel("Weighted F1 Score")
    ax.plot(means, color="lightgray", alpha=0.5)
    ax.grid()
    return fig, ax


def _plot_confusion_matrix(data, **params):
    fig, ax = plt.subplots()
    sns.heatmap(
        data,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        linewidths=1,
        linecolor="black",
        cbar=False,
        ax=ax
    )
    ax.set_xlabel("Actual Label")
    ax.set_ylabel("Predicted Label")
    ax.set_xticks([x + 0.5 for x in range(data.shape[0])], minor=True)
    ax.set_yticks([y + 0.5 for y in range(data.shape[1])], minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    return fig, ax


def plot_data(data, data_type, **params):
    # multiple classification plots
    if data_type == "loss":
        return _plot_loss(data, **params)
    elif data_type == "accuracy":
        return _plot_accuracy(data, **params)
    elif data_type == "weighted_f1":
        return _plot_weighted_f1(data, **params)
    elif data_type == "confusion_matrix":
        return _plot_confusion_matrix(data, **params)

