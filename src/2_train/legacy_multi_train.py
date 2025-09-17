from flow_package.multi_flow_env import MultiFlowEnv, InputType
import flow_package as f_p
import numpy as np
import os
import pandas as pd
from glob import glob
import gc
import seaborn as sns
import matplotlib
import random
from collections import deque, namedtuple
from itertools import count
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse
import pickle
import torch.backends.cudnn

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("start")

CONST = f_p.Const()

print("load data")
TRAIN_PATH = "data_cicids2017/3_final/cicids2017_scaled_after_gan.csv"
TEST_PATH = "data_cicids2017/1_formated/cicids2017_formated_test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
test_df = test_df[test_df["Attempted Category"] == -1]
test_df = normalize_data(test_df)

train_df = train_df.dropna(how="any").dropna(axis=1, how="any")
test_df = test_df.dropna(how="any").dropna(axis=1, how="any")

df = pd.concat([train_df, test_df])

labels = df["Label"].value_counts().index.tolist()
df["Label"] = df["Label"].map(lambda x: labels.index(x))

with open("label_list.txt", "w") as f:
    f.write(f"{len(labels)}\n")
    for label in labels:
        f.write(f"{label}\n")

train_df = df.iloc[:int(len(train_df))]
test_df = df.iloc[int(len(train_df)):]

print("check label length")
train_label_len = len(train_df["Label"].unique())
test_label_len = len(test_df["Label"].unique())

print(f"train_label_len: {train_label_len}, test_label_len: {test_label_len}")
if train_label_len != test_label_len:
    raise ValueError("train_label_len != test_label_len")

# buf_df = pd.DataFrame()
# for label in train_df["Label"].unique():
#     df_label = train_df[train_df["Label"] == label]
#     if len(df_label) > 10_000:
#         # clip 10_000
#         buf_df = pd.concat([buf_df, df_label.sample(10_000)])
#     else:
#         buf_df = pd.concat([buf_df, df_label])

# train_df = buf_df

print(train_df["Label"].value_counts())

train_input = InputType(
    data=train_df,
    sample_size=5_000,
    normalize_exclude_columns=["Protocol", "Destination Port"],
    exclude_columns=["Attempted Category"]
)
train_env = MultiFlowEnv(train_input)

test_input = InputType(
    data=test_df,
    sample_size=1_000,
    is_test=True,
    normalize_exclude_columns=["Protocol", "Destination Port"],
    exclude_columns=["Attempted Category"]
)
test_env = MultiFlowEnv(test_input)
print("load data done")

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

device_name = "cpu"
if True:
    if torch.cuda.is_available():
        device_name = "cuda:1"
    elif torch.mps.is_available():
        device_name = "mps"
    elif torch.mtia.is_available():
        device_name = "mtia"
    elif torch.xpu.is_available():
        device_name = "xpu"

device = torch.device(device_name)
print(f"device: {device_name}")

Transaction = namedtuple('Transaction', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transaction(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def moving_average(data, window_size):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def _plot_common_figure(fig=None, is_save=False, save_name=None, show_result=False):
    """
    グラフの共通処理（保存・表示・クリア）
    """
    if is_save and save_name:
        plt.savefig(f"result/multi/{save_name}")
        plt.close()
        return
    plt.pause(0.001)
    if 'is_ipython' in globals() and is_ipython:
        if not show_result:
            display.display(plt.gcf() if fig is None else fig)
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf() if fig is None else fig)


def plot_graph(data: list, show_result=False, is_save=False):
    """
    エピソードごとの比率推移グラフ
    """
    plt.figure(figsize=(15, 5))
    plt.title("Result" if show_result else "Training..." + f" {len(data)}")
    means = moving_average(data, 200)
    lines = np.full(len(means), 1 / 15)
    first = len(data) - len(means)
    if first < 0:
        first = 0
    else:
        means = np.concatenate((np.full(first, np.nan), means))
        lines = np.concatenate((np.full(first, np.nan), lines))
    plt.plot(lines, color="blue", linestyle="--")
    plt.xlabel("Episode")
    plt.ylabel("ratio")
    plt.plot(means, color="red")
    plt.grid()
    _plot_common_figure(is_save=is_save, save_name="graph.png", show_result=show_result)


def plot_normal_graph(data: list, show_result=False, is_save=False):
    """
    ステップごとの損失推移グラフ
    """
    plt.figure(figsize=(15, 5))
    plt.title("Result" if show_result else "Training..." + f" {len(data)}")
    means = moving_average(data, 200)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.plot(means, color="red")
    plt.grid()
    _plot_common_figure(is_save=is_save, save_name="normal_graph.png", show_result=show_result)


def plot_double_graph(data1: list, data2: list, show_result=False, is_save=False):
    """
    2つのグラフ（比率・損失）を並列表示
    """
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Result" if show_result else "Training..." + f" {len(data1)}")
    means_ax1 = moving_average(data1, 200)
    lines_ax1 = np.full(len(means_ax1), 1 / 15)
    first_ax1 = len(data1) - len(means_ax1)
    if first_ax1 < 0:
        first_ax1 = 0
    else:
        means_ax1 = np.concatenate((np.full(first_ax1, np.nan), means_ax1))
        lines_ax1 = np.concatenate((np.full(first_ax1, np.nan), lines_ax1))
    ax1.plot(lines_ax1, color="blue", linestyle="--")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("ratio")
    ax1.plot(means_ax1, color="red")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Result" if show_result else "Training..." + f" {len(data2)}")
    means_ax2 = moving_average(data2, 200)
    first_ax2 = len(data2) - len(means_ax2)
    if first_ax2 < 0:
        first_ax2 = 0
    else:
        means_ax2 = np.concatenate((np.full(first_ax2, np.nan), means_ax2))
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.plot(means_ax2, color="red")
    fig.tight_layout()
    _plot_common_figure(fig=fig, is_save=is_save, save_name="double_graph.png", show_result=show_result)


def plot_metrics(metrics_dict: dict, show_result=False, is_save=False):
    """
    複数の評価指標（accuracy, precision, recall, f1, fpr）をまとめて表示
    """
    fig = plt.figure(figsize=(16, 20))
    titles = ["Accuracy", "Precision", "Recall", "F1", "FPR"]
    colors = ["blue", "green", "red", "black", "purple"]
    keys = ["accuracy", "precision", "recall", "f1", "fpr"]
    for i, (title, color, key) in enumerate(zip(titles, colors, keys)):
        ax = fig.add_subplot(5, 1, i+1)
        ax.plot(metrics_dict[key], label=key, color=color)
        ax.grid()
        ax.set_title(title)
    plt.tight_layout()
    _plot_common_figure(fig=fig, is_save=is_save, save_name="metrics.png", show_result=show_result)


def calculate_metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if tp + fp != 0 else -1
    recall = tp / (tp + fn) if tp + fn != 0 else -1
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else None
    fpr = fp / (fp + tn) if fp + tn != 0 else None
    if precision < 0:
        precision = None
    if recall < 0:
        recall = None
    return accuracy, precision, recall, f1, fpr


def plot_confusion_matrix(confusion_array, class_names=None, show_result=False, is_save=False, name="confusion_matrix"):
    """
    混同行列のヒートマップ表示（グリッドをマス目に合わせ、割合を小数で表示、%記号なし）
    """
    plt.figure(figsize=(10, 8))
    plt.title("Result" if show_result else "Testing..." + f" {len(confusion_array)}")
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(confusion_array))]
    conf_matrix = confusion_array.copy().astype(float)
    for column in range(n_actions):
        sum_column = conf_matrix[:, column].sum()
        if sum_column == 0:
            continue
        conf_matrix[:, column] /= sum_column
    # ヒートマップ本体
    ax = sns.heatmap(
        conf_matrix,
        annot=True,
        cmap='Blues',
        fmt=".2f",  # 小数点2桁、%記号なし
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=1,  # マス目の線を太く
        linecolor='black',
        cbar=True
    )
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    # グリッド線をマス目に合わせて明示的に描画
    ax.set_xticks([x+0.5 for x in range(len(class_names))], minor=True)
    ax.set_yticks([y+0.5 for y in range(len(class_names))], minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    plt.grid(False)  # デフォルトのグリッドは消す
    _plot_common_figure(is_save=is_save, save_name=f"{name}.png", show_result=show_result)

def plot_macro_metrics(f1_list, precision_list, recall_list, is_save=True):
    """
    エピソードごとのマクロF1・マクロ精度・マクロリコールを1枚の線グラフで表示
    """
    plt.figure(figsize=(15, 5))
    plt.plot(f1_list, label='Macro F1', color='red')
    plt.plot(precision_list, label='Macro Precision', color='blue')
    plt.plot(recall_list, label='Macro Recall', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Macro Metrics per Episode' + f" {len(f1_list)}")
    plt.legend()
    plt.grid()
    _plot_common_figure(is_save=is_save, save_name="macro_metrics.png", show_result=True)

def plot_f1_stats(f1_max_list, f1_min_list, f1_mean_list, is_save=True):
    """
    エピソードごとのF1スコア最大・最小・平均の推移を1枚の折れ線グラフで表示
    """
    plt.figure(figsize=(15, 5))
    plt.plot(f1_max_list, label='F1 max', color='red')
    plt.plot(f1_min_list, label='F1 min', color='blue')
    plt.plot(f1_mean_list, label='F1 mean', color='green')
    plt.xlabel('Episode')
    plt.ylabel('F1 Score')
    plt.title('Per-class F1 Score (max/min/mean) per Episode' + f" {len(f1_max_list)}")
    plt.legend()
    plt.grid()
    _plot_common_figure(is_save=is_save, save_name="f1_stats.png", show_result=True)

PROTOCOL_DIM = 8
PORT_DIM = 32

MODEL_PATH = "multi_01.pth"

UPDATE_TARGET_STEPS = 200
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100000
TAU = 0.005
LR = 1e-4


class DeepFlowNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(DeepFlowNetwork, self).__init__()
        self.protocol_embedding = nn.Embedding(256, PROTOCOL_DIM)
        self.port_embedding = nn.Embedding(65536, PORT_DIM)
        n_inputs = n_inputs - 2 + PORT_DIM + PROTOCOL_DIM # - 2 + protocol + port
        self.fc1 = nn.Linear(n_inputs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_outputs)

    def forward(self, x):
        port_emb = self.port_embedding(x[0].long())
        protocol_emb = self.protocol_embedding(x[1].long())
        renew = torch.cat([port_emb, protocol_emb, x[2]], dim=1)
        renew = F.relu(self.fc1(renew))
        renew = F.relu(self.fc2(renew))
        return self.fc3(renew)


torch.backends.cudnn.benchmark = True  # ネットワーク構造が固定なら高速化

# optimize_modelのAMP対応
scaler = GradScaler('cuda') if device.type == 'cuda' else None

def get_reward(action, answer):
    if action == answer:
        return 1
    elif action == 0 or answer == 0:
        return -2
    else:
        return -1

num_episodes = 600
n_actions = train_env.action_space.n
n_inputs = train_env.observation_space.shape[0]

print(n_actions)

state = train_env.reset()

policy_net = DeepFlowNetwork(n_inputs, n_actions).to(device)
target_net = DeepFlowNetwork(n_inputs, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
steps_done = 0

memory = ReplayMemory(10000)
episode_rewards = []
episode_precision = []
loss_array = []
episode_macro_precision = []
episode_macro_recall = []
episode_macro_f1 = []
f1_max_list = []
f1_min_list = []
f1_mean_list = []

with open("result/multi/train.log", "w") as f:
    f.write("episode, show\n")

def _unpack_state_batch(batch_states):
    """
    状態バッチ（port, protocol, features）をまとめて展開
    """
    port = torch.cat([s[0] for s in batch_states])
    protocol = torch.cat([s[1] for s in batch_states])
    features = torch.cat([s[2] for s in batch_states])
    return [port, protocol, features]

def select_action(state_tensor: torch.Tensor):
    """
    ε-greedy法による行動選択
    """
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    # state_tensorはリスト形式（[port, protocol, other]）
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state_tensor).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long, device=device)

def optimize_model():
    """
    経験リプレイからバッチをサンプリングし、Qネットワークを最適化
    """
    if len(memory) < BATCH_SIZE:
        return None
    transitions = memory.sample(BATCH_SIZE)
    batch = Transaction(*zip(*transitions))
    state_batch = _unpack_state_batch(batch.state)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)
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

def test_model():
    """
    学習済みモデルのテストと混同行列の出力
    """
    trained_network = DeepFlowNetwork(n_inputs=n_inputs, n_outputs=n_actions).to(device)
    trained_network.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    trained_network.eval()
    confusion_array = np.zeros((n_actions, n_actions), dtype=np.int32)
    for i_loop in range(1):
        random.seed(i_loop)
        test_raw_state = test_env.reset()
        try:
            test_state = f_p.to_tensor(test_raw_state, device=device)
        except Exception:
            raise print(test_raw_state)
        for t in count():
            with torch.no_grad():
                test_action = trained_network(test_state).max(1).indices.view(1, 1)
            test_raw_next_state, test_reward, test_terminated, test_truncated, test_info = test_env.step(test_action.item())
            action = test_info["action"]
            answer = test_info["answer"]
            confusion_array[action, answer] += 1
            if test_terminated:
                break
            test_state = f_p.to_tensor(test_raw_next_state, device=device)
            if t % 100000 == 0:
                plot_confusion_matrix(confusion_array, is_save=True, name=f"test_confusion_matrix_{t}")
            if t % 20_000 == 0:
                print(f"\r==test step: {t:10d}/{len(test_df)}", end="")
    # 混同行列の内容をcsvと標準出力に出す
    # for i in range(n_actions):
    #     print("[", end="")
    #     for j in range(n_actions):
    #         if j == n_actions - 1:
    #             print(confusion_array[i, j], end="")
    #         else:
    #             print(confusion_array[i, j], end=", ")
    #     print("],")
    print(f"\r==test step: {len(test_df)}/{len(test_df)} done ==")
    with open("result/multi/test.csv", "w") as f:
        f.write("row,")
        for i in range(n_actions):
            f.write(f"{i},")
        else:
            f.write("\n")
        for i in range(n_actions):
            f.write(f"{i},")
            for j in range(n_actions):
                f.write(f"{confusion_array[i, j]},")
            f.write("\n")
    plot_confusion_matrix(confusion_array, show_result=True, is_save=True, name="test_confusion_matrix")

def train_model(num_episodes=100):
    """
    強化学習の訓練ループ本体
    """
    period = num_episodes // 100
    for i_episode in range(num_episodes):
        print(f"episode: {i_episode+1:3d}", end="")
        test = []
        random.seed(i_episode)
        sum_reward = 0
        confusion_matrix = np.zeros((n_actions, n_actions), dtype=int)
        initial_state = train_env.reset()
        state = f_p.to_tensor(initial_state, device=device)
        show = np.zeros(n_actions, dtype=int)
        for t in count():
            action = select_action(state)
            raw_next_state, reward, terminated, truncated, info = train_env.step(action.item())
            row_column_index = info["matrix_position"]
            confusion_matrix[row_column_index[0], row_column_index[1]] += 1
            test.append(info["answer"])
            show[action.item()] += 1
            reward = get_reward(info["action"], info["answer"])
            if terminated:
                with open("result/multi/train.log", "a") as f:
                    f.write(f"{i_episode}, {show}\n")
                next_state = None
            else:
                next_state = f_p.to_tensor(raw_next_state, device=device)
            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            memory.push(state, action, next_state, reward)
            sum_reward += reward.item() if reward.item() == 1 else 0
            state = next_state
            loss = optimize_model()
            if terminated:
                loss_array.append(loss)
                episode_rewards.append(sum_reward / (t + 1))
                break

        # confusion_matrixからy_true, y_predを作成
        y_true = []
        y_pred = []
        for pred in range(n_actions):
            for true in range(n_actions):
                count_buf = confusion_matrix[pred, true]
                y_true += [true] * count_buf
                y_pred += [pred] * count_buf

        macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        f1_max = f1_per_class.max()
        f1_min = f1_per_class.min()
        f1_mean = f1_per_class.mean()

        print(f", macro_precision: {macro_precision:.3f}, F1(max): {f1_max:.3f}, F1(mean): {f1_mean:.3f}, F1(min): {f1_min:.3f}")

        episode_macro_precision.append(macro_precision)
        episode_macro_recall.append(macro_recall)
        episode_macro_f1.append(macro_f1)
        f1_max_list.append(f1_max)
        f1_min_list.append(f1_min)
        f1_mean_list.append(f1_mean)

        # 5エピソードごとにモデル保存・可視化・テスト
        if i_episode % period == period - 1:
            plot_confusion_matrix(confusion_matrix, is_save=True, name=f"train_confusion_matrix")
            plot_macro_metrics(episode_macro_f1, episode_macro_precision, episode_macro_recall, is_save=True)
            torch.save(policy_net.state_dict(), MODEL_PATH)
            # plot_double_graph(episode_rewards, np.array(loss_array), is_save=True)

            test_model()

        if i_episode % 100 == 99:
            plot_f1_stats(f1_max_list, f1_min_list, f1_mean_list, is_save=True)
            plot_graph(episode_rewards, show_result=True, is_save=True)
            plot_normal_graph(loss_array, show_result=True, is_save=True)

        # 訓練終了後に指標履歴を保存
        history = {
            'macro_f1': episode_macro_f1,
            'macro_precision': episode_macro_precision,
            'macro_recall': episode_macro_recall,
            'rewards': episode_rewards,
            'loss': loss_array,
        }
        with open("result/multi/metrics_history.pkl", "wb") as f:
            pickle.dump(history, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='保存済みモデルから続きで訓練')
    args = parser.parse_args()

    # モデル・履歴のロード分岐
    if args.resume and os.path.exists(MODEL_PATH):
        print(f"== モデル {MODEL_PATH} をロードして続きから訓練します ==")
        policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        target_net.load_state_dict(policy_net.state_dict())
        if os.path.exists("result/multi/metrics_history.pkl"):
            with open("result/multi/metrics_history.pkl", "rb") as f:
                history = pickle.load(f)
                episode_macro_f1[:] = history.get('macro_f1', [])
                episode_macro_precision[:] = history.get('macro_precision', [])
                episode_macro_recall[:] = history.get('macro_recall', [])
                episode_rewards[:] = history.get('rewards', [])
                loss_array[:] = history.get('loss', [])
    else:
        print(f"== 新規で訓練を開始します ==")

    train_model(num_episodes)
    plot_macro_metrics(episode_macro_f1, episode_macro_precision, episode_macro_recall, is_save=True)
    plot_f1_stats(f1_max_list, f1_min_list, f1_mean_list, is_save=True)
    train_env.close()


