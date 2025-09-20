import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml
import pandas as pd
import numpy as np

from itertools import count
from glob import glob
from tqdm import tqdm

from flow_package import to_tensor
from flow_package.multi_flow_env import MultiFlowEnv, InputType

from lib.utils import setup_logging
from lib.network import DeepFlowNetwork


def load_csv(input):
    files = glob(os.path.join(input, "*.csv.gz"))
    df = pd.concat([
        pd.read_csv(f) for f in files
    ])
    return df


def write_result(cm_memory, prefix):
    # データを整数形式に変換して確実にする
    cm_data = []
    for predicted, actual in cm_memory:
        # テンソルの場合は.item()で値を取得、そうでなければそのまま使用
        pred_val = predicted.item() if hasattr(predicted, 'item') else int(predicted)
        actual_val = actual.item() if hasattr(actual, 'item') else int(actual)
        cm_data.append([pred_val, actual_val])
    
    cm = pd.DataFrame(cm_data, columns=["Predicted", "Actual"])
    cm_path = os.path.join("evaluate", f"{prefix}_confusion_matrix.csv")
    os.makedirs("evaluate", exist_ok=True)
    cm.to_csv(cm_path, index=False)
    # print(f"Confusion matrix saved to {cm_path}")


def test(df, params):
    input = InputType(
        data=df,
        is_test=True,
        normalize_exclude_columns=["Protocol", "Destination Port"],
        # exclude_columns=["Attempted Category"],
    )
    env = MultiFlowEnv(input)

    network = DeepFlowNetwork(
        n_inputs=env.observation_space.shape[0],
        n_outputs=env.action_space.n,
    )
    path = os.path.join("models", "multi_dqn_model.pth")
    network.load_state_dict(torch.load(path))
    network.eval()

    cm_memory = []

    log_path = os.path.join("logs", "evaluate.log")
    logger = setup_logging(log_path)
    logger.info("Starting evaluation...")

    for i_loop in range(1):
        raw_state = env.reset()
        try:
            state = to_tensor(raw_state)
        except Exception as e:
            raise ValueError(f"Error converting state to tensor: {e}")
        progress_bar = tqdm(range(len(df)), desc=f"Evaluation Loop {i_loop+1}")
        for t in count():
            with torch.no_grad():
                predicted_action = network(state).max(1).indices.view(1, 1)

            raw_next_state, reward, terminated, _, info = env.step(predicted_action.item())
            actual_action = info["action"]
            actual_answer = info["answer"]

            # 両方を整数形式で保存
            cm_memory.append([actual_action, actual_answer])

            if terminated:
                # logger.info(f"Evaluation finished after {t+1} steps")
                break
            try:
                next_state = to_tensor(raw_next_state)
            except Exception as e:
                raise ValueError(f"Error converting next state to tensor: {e}")

            state = next_state
            progress_bar.update(1)
    
        progress_bar.close()
    
    logger.info("Evaluation completed.")
    write_result(cm_memory, "evaluate")


def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <test_data_directory>")
        sys.exit(1)
    params = yaml.safe_load(open("params.yaml"))

    input = sys.argv[1]
    df = load_csv(input)

    test(df, params)


if __name__ == "__main__":
    main()