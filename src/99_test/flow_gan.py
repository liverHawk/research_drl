import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import pandas as pd
import numpy as np

from glob import glob

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler

from lib.gan import GAN


def dir_load_csv(dir_path):
    csv_files = glob(f"{dir_path}/*.csv.gz")
    print(f"Found {len(csv_files)} CSV files in {dir_path}")
    
    if len(csv_files) == 0:
        print(f"警告: {dir_path} にCSVファイルが見つかりません")
        return pd.DataFrame()
    
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            print(f"読み込み成功: {f} (行数: {len(df)})")
            dfs.append(df)
        except Exception as e:
            print(f"読み込み失敗: {f}, エラー: {e}")
    
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        print(f"総データ数: {len(df)}")
        return df
    else:
        return pd.DataFrame()


def load_csv(file_name):
    try:
        df = pd.read_csv(file_name)
        print(f"ファイル読み込み成功: {file_name} (行数: {len(df)})")
        return df
    except Exception as e:
        print(f"ファイル読み込み失敗: {file_name}, エラー: {e}")
        return pd.DataFrame()


class FlowDataset(Dataset):
    def __init__(self, dir_path=None, file_name=None):
        if dir_path is not None:
            self.data = dir_load_csv(dir_path)
        elif file_name is not None:
            self.data = load_csv(file_name)
        else:
            raise ValueError("dir_pathまたはfile_nameのいずれかを指定してください。")

        # データが空でないかチェック
        if self.data.empty:
            print("警告: データが空です")
            self.labels = pd.Series(dtype='object')
            return
        
        # カラム情報を表示
        print(f"利用可能なカラム: {list(self.data.columns)}")
        
        # Labelカラムが存在するかチェック
        if "Label" not in self.data.columns:
            print("警告: 'Label'カラムが見つかりません")
            # すべてのデータを使用し、ダミーラベルを作成
            self.labels = pd.Series([0] * len(self.data))
        else:
            # ラベルの分布を確認
            label_counts = self.data["Label"].value_counts()
            print(f"ラベル分布:")
            for label, count in label_counts.items():
                print(f"  {label}: {count}")
            
            # ラベル0のデータを抽出（正常データ）
            normal_data = self.data[self.data["Label"] == 0]
            if len(normal_data) == 0:
                print("警告: ラベル0のデータが見つかりません")
                # 最も多いラベルを使用
                most_common_label = label_counts.index[0]
                print(f"代わりにラベル'{most_common_label}'のデータを使用します")
                self.data = self.data[self.data["Label"] == most_common_label]
            else:
                self.data = normal_data
            
            self.labels = self.data["Label"]
            self.data = self.data.drop(columns=["Label"])
        
        # データを数値型に変換
        numeric_data = self.data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            print("警告: 数値データが見つかりません")
        else:
            self.data = numeric_data
            # NaNや無限値を処理
            self.data = self.data.fillna(0)
            self.data = self.data.replace([np.inf, -np.inf], 0)
            print(f"数値データの特徴量数: {self.data.shape[1]}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if len(self.data) == 0:
            raise IndexError("データセットが空です")
        
        # pandasのSeriesをTensorに変換
        data_values = self.data.iloc[idx].values.astype(np.float32)
        data_tensor = torch.from_numpy(data_values)
        
        label = self.labels.iloc[idx] if len(self.labels) > idx else 0
        
        # GANクラスとの互換性を保つため、シンプルにテンソルのみを返す
        return data_tensor


class FlowSampler(Sampler):
    def __init__(self, data, num_samples):
        self.data = data
        self.num_samples = num_samples
        
        if len(data) == 0:
            print("警告: データが空のため、サンプラーは空のイテレータを返します")

    def __iter__(self):
        if len(self.data) == 0:
            return iter([])
        
        # インデックスのリストを生成してイテレータとして返す
        indices = np.random.choice(
            len(self.data), 
            size=min(self.num_samples, len(self.data)), 
            replace=False
        )
        return iter(indices.tolist())

    def __len__(self):
        return min(self.num_samples, len(self.data))


def main():
    # データパスをより詳細にチェック
    path = os.path.abspath(os.path.join("./data/processed/normalized/train"))
    print(f"データ読み込み先: {path}")
    
    # パスの存在確認
    if not os.path.exists(path):
        print(f"エラー: パス '{path}' が存在しません")
        # 代替パスを試す
        alternative_paths = [
            "./data/train",
            "../data/processed/normalized/train",
            "../../data/processed/normalized/train"
        ]
        
        for alt_path in alternative_paths:
            alt_abs_path = os.path.abspath(alt_path)
            if os.path.exists(alt_abs_path):
                path = alt_abs_path
                print(f"代替パスを使用: {path}")
                break
        else:
            print("利用可能なデータパスが見つかりません")
            return
    
    dataset = FlowDataset(dir_path=path)
    
    # データサイズを確認
    print(f"データセットサイズ: {len(dataset)}")
    
    # データが空の場合の処理
    if len(dataset) == 0:
        print("エラー: データセットが空です。以下を確認してください:")
        print("1. データファイルが正しい場所にあるか")
        print("2. データファイルが正しい形式か")
        print("3. ラベル0（正常データ）が存在するか")
        return
    
    # サンプル数をデータサイズ以下に設定
    sample_size = min(1000, len(dataset))
    sampler = FlowSampler(dataset.data, num_samples=sample_size)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)

    # データ形状確認
    try:
        sample_batch = next(iter(dataloader))
        print(f"サンプルバッチ形状: {sample_batch.shape}")
        print(f"バッチサイズ: {sample_batch.shape[0]}")
        print(f"特徴量数: {sample_batch.shape[1]}")
        print(f"データ型: {sample_batch.dtype}")

        print(sample_batch)

        # GANの設定
        gan = GAN(
            lr=0.0002,
            d_input_size=sample_batch.shape[1],  # 特徴量数
            g_output_size=sample_batch.shape[1],  # 特徴量数
            noise_size=100,  # ノイズサイズも調整
            loss_type='wgan-gp',  # より安定したロス関数を使用
            save=False
        )
        
        # データローダーを設定
        gan.dataloader = dataloader

        print("GANの学習を開始します...")
        gan.train(epochs=100)
        
        print("サンプル生成を開始します...")
        gan.generate()

        
    except StopIteration:
        print("エラー: データローダーからサンプルを取得できません")
    except Exception as e:
        print(f"予期しないエラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
