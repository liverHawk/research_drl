import os
import yaml
import sys

# Add the src directory to the Python path to access lib module
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from ipaddress import ip_address as ip
from sklearn.preprocessing import LabelEncoder
import mlflow
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from lib.utils import CICIDS2017, BASE, setup_logging
from lib.csv import save_split_csv, multiprocess_save_csv, save_csv


def adjust_labels(df, adjustment):
    labels = df["Label"].unique()

    # attempted: false -> change benign
    # combine_web_attack: true -> combine Web Attack
    # tool_separate: false -> combine DoS tools
    params = [
        adjustment["attempted"] is False,
        adjustment["combine_web_attack"] is True,
        adjustment["tool_separate"] is False,
    ]
    print(params)

    for label in labels:
        if params[0] and "Attempted" in label:
            df.loc[df["Label"] == label, "Label"] = "BENIGN"
            continue
        if params[1] and "Web Attack" in label:
            df.loc[df["Label"] == label, "Label"] = "Web Attack"
            continue
        if params[2] and "DoS " in label:
            df.loc[df["Label"] == label, "Label"] = "DoS"
            continue
        if "Infiltration" in label:
            df.loc[df["Label"] == label, "Label"] = "Infiltration"
    
    return df


def fast_process(df, type="normal"):
    if type == "normal":
        df = df.drop(CICIDS2017().get_delete_columns(), axis=1)
        df = df.drop(columns=['Attempted Category'])
    elif type == "full":
        df = df.drop(['Flow ID', 'Src IP', 'Attempted Category'], axis=1)
        # Timestamp→秒
        df['Timestamp'] = (
            pd.to_datetime(df['Timestamp'], format="%Y-%m-%d %H:%M:%S.%f")
            .astype('int64') // 10**9
        )
        # IP文字列→整数
        df['Dst IP'] = df['Dst IP'].apply(lambda x: int(ip.IPv4Address(x)))
    # 欠損／無限大落とし
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def min_label_count(df):
    """
    Returns the minimum count of labels in the DataFrame.
    """
    return df["Label"].value_counts().min()


def data_process(input_path, params):
    log_path = os.path.join("logs", "cleaning.log")
    logger = setup_logging(log_path)

    logger.info("Starting data processing...")
    files = glob(f"{input_path}/*.csv")
    dfs = [fast_process(pd.read_csv(f)) for f in tqdm(files)]
    df = pd.concat(dfs, ignore_index=True)

    df = adjust_labels(df, params["adjustment"])
    df = df.dropna().dropna(axis=1, how='all')

    rename_dict = {
        k: v for k, v in zip(
            CICIDS2017().get_features_labels(),
            BASE().get_features_labels()
        )
    }
    df = df.rename(columns=rename_dict)

    logger.info("Label encoding...")
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["Label"])

    with open("labels.txt", "w") as f:
        for class_label in le.classes_:
            f.write(f"{class_label}\n")

    return df


def main():
    all_params = yaml.safe_load(open("params.yaml"))
    params = all_params["prepare"]

    if all_params["mlflow"]["use_azure"]:
        path = os.path.join(os.path.dirname(__file__), "..", "..", "config.json")
        ml_client = MLClient.from_config(credential=DefaultAzureCredential(), config_path=path)
        mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    else:
        mlflow_tracking_uri = all_params["mlflow"]["tracking_uri"]
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(
        f"{all_params['mlflow']['experiment_prefix']}_cleaning"
    )
    mlflow.start_run()
    mlflow.log_param("mlflow_tracking_uri", mlflow_tracking_uri)

    print(params)

    if len(sys.argv) != 2:
        print("Usage: python src/prepare.py <input_file_directory>")
        sys.exit(1)

    input = sys.argv[1]
    if not os.path.exists(input):
        print(f"Input file {input} does not exist.")
        sys.exit(1)

    prepare_dir = os.path.join("data", "processed", "cleaned")
    os.makedirs(os.path.join("logs"), exist_ok=True)
    os.makedirs(prepare_dir, exist_ok=True)
    os.makedirs(os.path.join(prepare_dir, "clean"), exist_ok=True)

    dataset_path = input

    df = data_process(
        input_path=dataset_path,
        params=params
    )

    files = save_split_csv(
        df=df,
        output_dir=prepare_dir,
        output_prefix="cleaned",
        chunk_size=500_000
    )

    try:
        multiprocess_save_csv(
            dfs=[df for df, _ in files],
            paths=[path for _, path in files]
        )
    except Exception as e:
        print(f"Error during saving CSV files: {e}")
        for file in files:
            save_csv(file[0], file[1])

    mlflow.log_artifact("logs/cleaning.log")
    mlflow.log_artifact("labels.txt")

    mlflow.end_run()


if __name__ == "__main__":
    main()
