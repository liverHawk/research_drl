import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def moving_average(data, window_size):
    if window_size <= 1 or len(data) == 0:
        return np.array(data)
    if len(data) < window_size:
        window_size = max(1, len(data))
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def _plot_loss(data, **params):
    fig, ax = plt.subplots()
    ax.set_title(("Result" if params.get("show_result") else "Training...") + f" {len(data)}")
    window = params.get("window", 200)
    if len(data) == 0:
        means = np.array([])
    else:
        w = min(window, max(1, len(data)))
        means = moving_average(data, w)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    if means.size:
        ax.plot(np.arange(len(means)), means, color="lightgray", alpha=0.8)
    ax.grid()
    save_path = params.get("save_path")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def _plot_accuracy(data, **params):
    fig, ax = plt.subplots()
    ax.set_title(("Result" if params.get("show_result") else "Training...") + f" {len(data)}")
    window = params.get("window", 200)
    if len(data) == 0:
        means = np.array([])
    else:
        w = min(window, max(1, len(data)))
        means = moving_average(data, w)
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    if means.size:
        ax.plot(np.arange(len(means)), means, color="lightgray", alpha=0.8)
    ax.grid()
    save_path = params.get("save_path")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def _plot_weighted_f1(data, **params):
    fig, ax = plt.subplots()
    ax.set_title(("Result" if params.get("show_result") else "Training...") + f" {len(data)}")
    window = params.get("window", 200)
    if len(data) == 0:
        means = np.array([])
    else:
        w = min(window, max(1, len(data)))
        means = moving_average(data, w)
    ax.set_xlabel("Step")
    ax.set_ylabel("Weighted F1 Score")
    if means.size:
        ax.plot(np.arange(len(means)), means, color="lightgray", alpha=0.8)
    ax.grid()
    save_path = params.get("save_path")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def _plot_confusion_matrix(data, **params):
    fig, ax = plt.subplots(figsize=params.get("figsize", (6, 5)))
    sns.heatmap(
        data,
        annot=True,
        cmap="Blues",
        fmt=params.get("fmt", ".2f"),
        linewidths=1,
        linecolor="black",
        cbar=False,
        ax=ax
    )
    ax.set_xlabel("Actual Label")
    ax.set_ylabel("Predicted Label")
    # remove the minor-tick grid approach if matrix is integer-shaped
    try:
        ax.set_xticks([x + 0.5 for x in range(data.shape[1])], minor=True)
        ax.set_yticks([y + 0.5 for y in range(data.shape[0])], minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    except Exception:
        pass
    save_path = params.get("save_path")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
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

