import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

from itertools import count
from glob import glob
from tqdm import tqdm

from flow_package import to_tensor
from flow_package.multi_flow_env import MultiFlowEnv, InputType

from lib.utils import setup_logging
from lib.network import DeepFlowNetwork

# 追加インポート: mlflow とプロットユーティリティ
import mlflow
from lib import plot as plot_lib


def load_csv(input):
    files = glob(os.path.join(input, "*.csv.gz"))
    df = pd.concat([
        pd.read_csv(f) for f in files
    ])
    return df

def write_result(cm_true, cm_pred, episode, n_input, n_output):
    # 混同行列の計算
    cm = confusion_matrix(cm_true, cm_pred, labels=range(n_output), normalize="true")
    with open("logs/evaluate_result.log", "a") as f:
        f.write(f"Episode {episode}:\n")
        f.write(f"{cm}\n")
    total = cm.sum()
    accuracy = float(cm.diagonal().sum() / total) if total > 0 else 0.0
    return cm, accuracy

# def write_result(cm_memory, prefix):
#     # cm_memory: list of [predicted, actual]
#     cm_data = []
#     for pred, actual in cm_memory:
#         pred_val = pred.item() if hasattr(pred, "item") else int(pred)
#         actual_val = actual.item() if hasattr(actual, "item") else int(actual)
#         cm_data.append([pred_val, actual_val])

#     if len(cm_data) == 0:
#         # 空の場合は空のファイルを作るだけ
#         os.makedirs("evaluate", exist_ok=True)
#         cm_path = os.path.join("evaluate", f"{prefix}_confusion_matrix.csv")
#         pd.DataFrame(columns=["Predicted", "Actual"]).to_csv(cm_path, index=False)
#         return None, 0.0, cm_path

#     # クラス数はデータ上の最大ラベルから推定
#     preds = [p for p, a in cm_data]
#     actuals = [a for p, a in cm_data]
#     n = max(max(preds), max(actuals)) + 1
#     cm = np.zeros((n, n), dtype=int)

#     # plot.py のラベル付け（x: Actual, y: Predicted）に合わせて cm[predicted][actual] を増やす
#     for pred, actual in cm_data:
#         cm[pred][actual] += 1

#     os.makedirs("evaluate", exist_ok=True)
#     cm_path = os.path.join("evaluate", f"{prefix}_confusion_matrix.csv")
#     # CSV 保存（行: Predicted, 列: Actual のマトリクス形式）
#     pd.DataFrame(cm).to_csv(cm_path, index=True)

#     total = cm.sum()
#     accuracy = float(np.trace(cm) / total) if total > 0 else 0.0
#     return cm, accuracy, cm_path


def test(df, params):
    mlflow.autolog()
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
            # predicted と actual を混同行列用に保存（predicted, actual）
            actual_answer = info["answer"]
            cm_memory.append([predicted_action, actual_answer])

            if terminated:
                break
            try:
                next_state = to_tensor(raw_next_state)
            except Exception as e:
                raise ValueError(f"Error converting next state to tensor: {e}")

            state = next_state
            progress_bar.update(1)

        progress_bar.close()

    logger.info("Evaluation completed.")

    # 結果保存と mlflow ログ
    cm, accuracy, cm_csv_path = write_result(cm_memory, "evaluate")
    # 数値メトリクスを mlflow に保存（数値のみ）
    total_samples = int(cm.sum()) if cm is not None else 0
    mlflow.log_metric("accuracy", float(accuracy))
    mlflow.log_metric("total_samples", int(total_samples))

    # 混同行列をプロットして保存・アップロード
    os.makedirs("evaluate/plots", exist_ok=True)
    if cm is not None:
        cm_img_path = os.path.join("evaluate/plots", "confusion_matrix_evaluate.png")
        plot_lib.plot_data(cm, "confusion_matrix", save_path=cm_img_path, fmt=".0f")
        mlflow.log_artifact(cm_img_path, artifact_path="evaluate/plots")

    # CSV もアーティファクトとして保存
    mlflow.log_artifact(cm_csv_path, artifact_path="evaluate")

    logger.info(f"Saved evaluation artifacts. accuracy={accuracy:.6f}, samples={total_samples}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <test_data_directory>")
        sys.exit(1)
    params = yaml.safe_load(open("params.yaml"))

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(
        f"{params['mlflow']['experiment_prefix']}_evaluate"
    )
    mlflow.start_run()

    input = sys.argv[1]
    df = load_csv(input)

    test(df, params)
    mlflow.end_run()


if __name__ == "__main__":
    main()