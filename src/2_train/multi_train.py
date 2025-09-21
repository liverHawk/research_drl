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

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from itertools import count
from glob import glob
from torch.amp import autocast, GradScaler
from tqdm import tqdm

import flow_package as fp
from flow_package.multi_flow_env import MultiFlowEnv, InputType

from lib.utils import setup_logging
from lib.network import DeepFlowNetwork
from lib.deep_learn import ReplayMemory, moving_average, Transaction

# Global variable for epsilon-greedy action selection
steps_done = 0


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
            # device=device
        )


def optimize_model(kwargs):
    BATCH_SIZE = kwargs.get("BATCH_SIZE", 128)
    GAMMA = kwargs.get("GAMMA", 0.999)
    device = kwargs.get("device", torch.device("cpu"))
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
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat([torch.tensor([rew], dtype=torch.float32) for rew in batch.reward]).to(device)
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool)
    non_final_next_states = [s for s in batch.next_state if s is not None]
    if non_final_next_states:
        next_state_batch = _unpack_state_batch(non_final_next_states)
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
        "device": None,
        "memory": None,
        "policy_net": None,
        "target_net": None,
        "optimizer": None,
        "scaler": GradScaler()
    }
    return select_args, optimize_args


def write_result(cm_memory, episode, n_input, n_output):
    # 混同行列の計算
    cm = np.zeros((n_input, n_output), dtype=int)
    for action, answer in cm_memory:
        cm[action][answer] += 1
    with open("logs/train_result.log", "a") as f:
        f.write(f"Episode {episode}:\n")
        f.write(f"{cm}\n")


def train(df, params):
    logger = setup_logging("logs/train.log")

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
    policy_net = DeepFlowNetwork(n_states, n_actions)
    target_net = DeepFlowNetwork(n_states, n_actions)
    target_net.load_state_dict(policy_net.state_dict())

    cm_memory = []

    select_args, optimize_args = get_args(params)
    optimizer = optim.Adam(policy_net.parameters(), lr=params["lr"])
    memory = ReplayMemory(params["memory_size"])

    select_args["policy_net"] = policy_net
    select_args["n_actions"] = n_actions

    optimize_args["device"] = torch.device(params.get("device", "cpu"))
    optimize_args["memory"] = memory
    optimize_args["policy_net"] = policy_net
    optimize_args["target_net"] = target_net
    optimize_args["optimizer"] = optimizer

    for i_episode in tqdm(range(params["n_episodes"])):
        random.seed(i_episode)

        initial_state = env.reset()
        state = fp.to_tensor(initial_state)

        for t in count():
            action = select_action(state, **select_args)
            raw_next_state, reward, terminated, _, info = env.step(action.item())

            # Ensure reward is a scalar by summing if it's an array/list
            if isinstance(reward, (list, tuple, np.ndarray)):
                reward = np.sum(reward)

            cm_memory.append([
                info["action"],
                info["answer"],
            ])

            next_state = fp.to_tensor(raw_next_state) if not terminated else None
            memory.push(state, action, next_state, reward)
            state = next_state

            loss = optimize_model(optimize_args)
            if terminated:
                # logger.info(f"Episode {i_episode} finished after {t+1} steps")
                break

        write_result(cm_memory, i_episode, n_states, n_actions)
        cm_memory = []
    
    path = os.path.join("models", "multi_dqn_model.pth")
    os.makedirs("models", exist_ok=True)
    torch.save(policy_net.state_dict(), path)


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/train/multi_train.py <input_file_directory>")
        sys.exit(1)
    input = sys.argv[1]
    params = yaml.safe_load(open("params.yaml"))["train"]

    with open("logs/train_result.log", "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")

    df = load_csv(input)

    train(df, params)


if __name__ == "__main__":
    main()