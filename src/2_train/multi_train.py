import os
import sys
import yaml
import random
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
import mlflow


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from itertools import count
from glob import glob
from torch.amp import autocast, GradScaler
from tqdm import tqdm

import flow_package as fp
from flow_package.multi_flow_env import MultiFlowEnv, InputType

from lib import plot as plot_lib
from lib.utils import setup_logging
from lib.network import DeepFlowNetwork
from lib.deep_learn import ReplayMemory, moving_average, Transaction

# Global variable for epsilon-greedy action selection
steps_done = 0

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# def to_tensor(x, device=torch.device("cpu")):
#     # numpy配列を効率的にテンソルに変換
#     if isinstance(x, (list, tuple)):
#         x = np.array(x)
    
#     # ポートとプロトコルは整数型、バッチサイズ1で作成
#     port = torch.tensor([x[0]], dtype=torch.long, device=device)
#     protocol = torch.tensor([x[1]], dtype=torch.long, device=device)
#     # 残りの特徴量は浮動小数点型、バッチサイズ1で作成
#     features = torch.tensor(x[2:].reshape(1, -1), dtype=torch.float32, device=device)
#     return [port, protocol, features]

def load_csv(input):
    files = glob(os.path.join(input, "*.csv.gz"))
    df = pd.concat([
        pd.read_csv(f) for f in files
    ])
    return df


def _unpack_state_batch(state_batch):
    port = torch.cat([s[0] for s in state_batch])
    protocol = torch.cat([s[1] for s in state_batch])
    features = torch.cat([s[2] for s in state_batch])
    return [port, protocol, features]


def select_action(state_tensor: torch.Tensor, **kwargs):
    EPS_END = kwargs.get("EPS_END", 0.05)
    EPS_START = kwargs.get("EPS_START", 0.9)
    EPS_DECAY = kwargs.get("EPS_DECAY", 200)
    policy_net = kwargs.get("policy_net")
    n_actions = kwargs.get("n_actions")
    steps_done = kwargs.get("steps_done", 0)

    """
    ε-greedy法による行動選択
    """
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    # state_tensorはリスト形式（[port, protocol, other]）
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state_tensor).max(1).indices.view(1, 1)
    else:
        return torch.tensor(
            [[random.randrange(n_actions)]],
            dtype=torch.long,
            device=device
        )


def optimize_model(kwargs):
    BATCH_SIZE = kwargs.get("BATCH_SIZE", 128)
    GAMMA = kwargs.get("GAMMA", 0.999)
    # device = kwargs.get("device", torch.device("cpu"))
    memory = kwargs.get("memory")
    policy_net = kwargs.get("policy_net")
    target_net = kwargs.get("target_net")
    optimizer = kwargs.get("optimizer")
    scaler = kwargs.get("scaler", None)
    """
    経験リプレイからバッチをサンプリングし、Qネットワークを最適化
    """
    if len(memory) < BATCH_SIZE:
        return None
    transitions = memory.sample(BATCH_SIZE)
    batch = Transaction(*zip(*transitions))
    state_batch = _unpack_state_batch(batch.state)
    # move state tensors to device (state_batch is a list of tensors)
    try:
        state_batch = [s.to(device) for s in state_batch]
    except Exception:
        pass
    # # state_batch は [port, protocol, features] のリスト -> 各テンソルを device へ
    # try:
    #     state_batch = [s.to(device) for s in state_batch]
    # except Exception:
    #     # 既に device 上にあるか、非テンソルが入っている場合はそのままにする
    #     pass
    # actions should be LongTensor with shape (BATCH_SIZE, 1)
    action_batch = torch.cat(batch.action).to(device).long()
    # rewards: ensure each stored reward is a 1-d tensor; build a (BATCH_SIZE,) float tensor
    try:
        reward_batch = torch.cat([r.view(-1) for r in batch.reward]).to(device).float()
    except Exception:
        # fallback: convert elements to floats then to tensor
        reward_batch = torch.tensor([float(r) for r in batch.reward], dtype=torch.float32, device=device)
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool)
    non_final_next_states = [s for s in batch.next_state if s is not None]
    if non_final_next_states:
        next_state_batch = _unpack_state_batch(non_final_next_states)
        # move next_state tensors to device
        try:
            next_state_batch = [s.to(device) for s in next_state_batch]
        except Exception:
            pass
        # try:
        #     next_state_batch = [s.to(device) for s in next_state_batch]
        # except Exception:
        #     pass
    else:
        next_state_batch = None
    if scaler is not None:
        with autocast(device_type=device.type):
            state_action_values = policy_net(state_batch).gather(1, action_batch)
            next_state_values = torch.zeros(BATCH_SIZE, device=device)
            if next_state_batch is not None:
                with torch.no_grad():
                    # Half→Float32にキャストして代入
                    next_state_values[non_final_mask] = target_net(next_state_batch).max(1).values.float()
            expected_state_action_values = reward_batch + GAMMA * next_state_values
            loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        utils.clip_grad_value_(policy_net.parameters(), 1000)
        scaler.step(optimizer)
        scaler.update()
    else:
        state_action_values = policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        if next_state_batch is not None:
            with torch.no_grad():
                next_state_values[non_final_mask] = target_net(next_state_batch).max(1).values
        expected_state_action_values = reward_batch + GAMMA * next_state_values
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(policy_net.parameters(), 1000)
        optimizer.step()
    return loss.item()


def get_args(params):
    select_args = {
        "EPS_END": params.get("eps_end", 0.05),
        "EPS_START": params.get("eps_start", 0.9),
        "EPS_DECAY": params.get("eps_decay", 200),
        "policy_net": None,
        "n_actions": None
    }
    optimize_args = {
        "BATCH_SIZE": params.get("batch_size", 128),
        "GAMMA": params.get("gamma", 0.999),
        "memory": None,
        "policy_net": None,
        "target_net": None,
        "optimizer": None,
        "scaler": GradScaler()
    }
    return select_args, optimize_args


def write_result(cm_memory, episode, n_input, n_output):
    # 混同行列の計算
    cm = np.zeros((n_output, n_output), dtype=int)
    for action, answer in cm_memory:
        cm[action][answer] += 1
    with open("logs/train_result.log", "a") as f:
        f.write(f"Episode {episode}:\n")
        f.write(f"{cm}\n")
    total = cm.sum()
    accuracy = float(cm.diagonal().sum() / total) if total > 0 else 0.0
    return cm, accuracy


def train(df, params):
    logger = setup_logging("logs/train.log")

    # ログパラメータを保存
    mlflow.log_params({
        "n_episodes": params.get("n_episodes"),
        "batch_size": params.get("batch_size"),
        "lr": params.get("lr"),
        "memory_size": params.get("memory_size")
    })

    logger.info("Starting training...")
    label_count = len(df["Label"].unique())
    reward_matrix = np.ones((label_count, label_count)) * -1.0
    np.fill_diagonal(reward_matrix, 1.0)

    columns = df.columns.tolist()
    columns.remove("Label")

    input = InputType(
        data=df,
        sample_size=params.get("sample_size", 100000),
        normalize_exclude_columns=["Protocol", "Destination Port"],
        # exclude_columns=["Attempted Category"],
        reward_list=reward_matrix
    )
    env = MultiFlowEnv(input)

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    logger.info(f"State space: {n_states}, Action space: {n_actions}")
    policy_net = DeepFlowNetwork(n_states, n_actions).to(device)
    target_net = DeepFlowNetwork(n_states, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    cm_memory = []

    # メトリクス収集用
    metrics = {
        "reward": [],
        "loss": [],
        "accuracy": [],
        "steps": [],
        "last_cm": None
    }

    select_args, optimize_args = get_args(params)
    optimizer = optim.Adam(policy_net.parameters(), lr=params["lr"])
    memory = ReplayMemory(params["memory_size"])

    select_args["policy_net"] = policy_net
    select_args["n_actions"] = n_actions

    optimize_args["memory"] = memory
    optimize_args["policy_net"] = policy_net
    optimize_args["target_net"] = target_net
    optimize_args["optimizer"] = optimizer

    os.makedirs("train/plots", exist_ok=True)

    log_plot_interval = params.get("log_plot_interval", 10)

    for i_episode in tqdm(range(params["n_episodes"])):
        random.seed(i_episode)

        initial_state = env.reset()
        state = fp.to_tensor(initial_state, device=device)

        episode_reward = 0.0
        episode_losses = []
        episode_steps = 0

        for t in tqdm(count(), leave=False):
            action = select_action(state, **select_args)
            raw_next_state, reward, terminated, _, info = env.step(action.item())

            # Ensure reward is a scalar by summing if it's an array/list
            if isinstance(reward, (list, tuple, np.ndarray)):
                reward = np.sum(reward)
            episode_reward += float(reward)
            # store a 1-d scalar tensor for reward (shape: [1]) to keep consistent shapes
            preserve_reward = torch.tensor([float(reward)], dtype=torch.float32, device=device)

            cm_memory.append([
                info["action"],
                info["answer"],
            ])

            next_state = fp.to_tensor(raw_next_state, device=device) if not terminated else None
            memory.push(state, action, next_state, preserve_reward)
            state = next_state

            loss = optimize_model(optimize_args)
            if loss is not None:
                episode_losses.append(loss)
            episode_steps += 1

            if terminated:
                break

        avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        cm, accuracy = write_result(cm_memory, i_episode, n_states, n_actions)

        metrics["reward"].append(episode_reward)
        metrics["loss"].append(avg_loss)
        metrics["accuracy"].append(accuracy)
        metrics["steps"].append(episode_steps)
        metrics["last_cm"] = cm.copy()

        # エピソードごとにログ用CSVへ追記
        os.makedirs("train", exist_ok=True)
        metrics_df = pd.DataFrame({
            "episode": np.arange(len(metrics["reward"])),
            "reward": metrics["reward"],
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "steps": metrics["steps"]
        })
        metrics_df.to_csv("train/metrics.csv", index=False)

        # mlflow に逐次メトリクスを送信（step=i_episode）
        mlflow.log_metric("reward", float(episode_reward), step=i_episode)
        mlflow.log_metric("loss", float(avg_loss), step=i_episode)
        mlflow.log_metric("accuracy", float(accuracy), step=i_episode)
        mlflow.log_metric("steps", int(episode_steps), step=i_episode)

        # プロットを定期的に作成して mlflow にアップロード
        if (i_episode + 1) % log_plot_interval == 0 or i_episode == params["n_episodes"] - 1:
            # Loss plot
            loss_path = os.path.join("train/plots", f"loss_ep_{i_episode:04d}.png")
            plot_lib.plot_data(metrics["loss"], "loss", save_path=loss_path, window=params.get("plot_window", 50))
            mlflow.log_artifact(loss_path)

            # Accuracy plot
            acc_path = os.path.join("train/plots", f"accuracy_ep_{i_episode:04d}.png")
            plot_lib.plot_data(metrics["accuracy"], "accuracy", save_path=acc_path, window=params.get("plot_window", 50))
            mlflow.log_artifact(acc_path)

            # 最後の混同行列（存在する場合）
            if metrics["last_cm"] is not None:
                cm_path = os.path.join("train/plots", f"confusion_ep_{i_episode:04d}.png")
                plot_lib.plot_data(metrics["last_cm"], "confusion_matrix", save_path=cm_path, fmt=".0f")
                mlflow.log_artifact(cm_path)

        cm_memory = []

    # 学習終了後に最終アーティファクトを保存
    # モデル
    path = os.path.join("models", "multi_dqn_model.pth")
    os.makedirs("models", exist_ok=True)
    torch.save(policy_net.state_dict(), path)
    mlflow.log_artifact(path, artifact_path="models")

    # 全メトリクスCSV と logs をまとめて保存
    mlflow.log_artifact("train/metrics.csv")
    mlflow.log_artifacts("train/plots")


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/train/multi_train.py <input_file_directory>")
        sys.exit(1)
    input = sys.argv[1]
    all_params = yaml.safe_load(open("params.yaml"))

    mlflow.set_tracking_uri(all_params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(
        f"{all_params['mlflow']['experiment_prefix']}_train"
    )
    params = all_params["train"]

    with open("logs/train_result.log", "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")

    mlflow.start_run()
    df = load_csv(input)

    train(df, params)
    mlflow.end_run()


if __name__ == "__main__":
    main()
