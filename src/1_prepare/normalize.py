import os
from glob import glob
import sys
import yaml
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import mlflow

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from lib.utils import setup_logging
from lib.csv import save_split_csv, multiprocess_save_csv, save_csv


def normalize(df, method, logger):
    logger.info(f"Normalizing data using {method}...")

    columns = df.columns
    if "Label" in columns:
        columns = columns.drop("Label")
    
    if method == "MinMaxScaler":
        for col in columns:
            min_val = df[col].min()
            max_val = df[col].max()
            logger.info(f" - {col}: min={min_val}, max={max_val}")
            df[col] = (df[col] - min_val) / (max_val - min_val)
        return df
    elif method == "StandardScaler":
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
        return df
    else:
        return df


def save_df(train_df, test_df, path, logger):
    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "test")

    train = save_split_csv(
        df=train_df,
        output_dir=train_path,
        output_prefix="train_normalized",
        chunk_size=500_000
    )
    test = save_split_csv(
        df=test_df,
        output_dir=test_path,
        output_prefix="test_normalized",
        chunk_size=500_000
    )
    files = train + test

    try:
        multiprocess_save_csv(
            dfs=[df for df, _ in files],
            paths=[path for _, path in files]
        )
    except Exception as e:
        logger.error(f"Error during saving CSV files: {e}")
        for file in files:
            save_csv(file[0], file[1])
        raise e


def main():
    if len(sys.argv) != 2:
        print("Usage: python normalize.py <data_path>")
        sys.exit(1)

    data_path = sys.argv[1]
    all_params = yaml.safe_load(open("params.yaml"))

    mlflow.set_tracking_uri(all_params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(
        f"{all_params['mlflow']['experiment_prefix']}_normalize"
    )

    params = all_params["prepare"]

    log_path = os.path.join("logs", "normalize.log")
    logger = setup_logging(log_path)

    mlflow.start_run()

    files = glob(os.path.join(data_path, "*.csv.gz"))
    dfs = [
        pd.read_csv(f) for f in tqdm(files)
    ]
    df = pd.concat(dfs, ignore_index=True)

    df = normalize(df, params["normalize_method"], logger)

    normalize_path = os.path.join(data_path, "..", "normalized")
    os.makedirs(normalize_path, exist_ok=True)
    os.makedirs(os.path.join(normalize_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(normalize_path, "test"), exist_ok=True)

    train_df, test_df = train_test_split(
        df, test_size=params["test_size"], random_state=42, stratify=df["Label"]
    )

    save_df(train_df, test_df, normalize_path, logger)

    mlflow.log_artifact(str(log_path))
    mlflow.end_run()


if __name__ == "__main__":
    main()