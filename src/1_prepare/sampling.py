import os
from glob import glob
from tqdm import tqdm
import sys
import yaml
import imblearn.over_sampling as imblearn_os
import pandas as pd
import mlflow
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from lib.utils import setup_logging
from lib.csv import save_split_csv, multiprocess_save_csv, save_csv


def min_label_count(df):
    """
    Returns the minimum count of labels in the DataFrame.
    """
    return df["Label"].value_counts().min()


def oversampling(df, method, method_params):
    min_count = min_label_count(df)
    if min_count + 1 < method_params["neighbors"]:
        raise ValueError(
            f"Minimum label count {min_count} "
            f"is less than neighbors {method_params['neighbors']}"
        )

    if method == "None":
        return df
    elif method == "SMOTE":
        os_method = imblearn_os.SMOTE(
            k_neighbors=method_params["neighbors"],
            sampling_strategy=method_params["sampling_strategy"],
            random_state=method_params["seed"]
        )
    elif method == "ADASYN":
        os_method = imblearn_os.ADASYN(
            n_neighbors=method_params["neighbors"],
            sampling_strategy=method_params["sampling_strategy"],
            random_state=method_params["seed"]
        )
    else:
        raise ValueError(f"Unsupported oversampling method: {method}")

    X = df.drop("Label", axis=1)
    y = df["Label"]

    X_resampled, y_resampled = os_method.fit_resample(X, y)

    return pd.concat([X_resampled, y_resampled], axis=1)

# tenant dc6b9ce9-a6ba-4ed3-bd67-9f9e3ac85cf1
# client id
# client secret
def main():
    logger = setup_logging("logs/sampling.log")
    FIXED_ROWS = 500_000
    if len(sys.argv) != 2:
        logger.error("Usage: python sampling.py <data_path>")
        sys.exit(1)
    all_params = yaml.safe_load(open("params.yaml"))

    if all_params["mlflow"]["use_azure"]:
        ml_client = MLClient.from_config(credential=DefaultAzureCredential(), config_path="./config.json")
        mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    else:
        mlflow_tracking_uri = all_params["mlflow"]["tracking_uri"]
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(
        f"{all_params['mlflow']['experiment_prefix']}_sampling"
    )
    mlflow.start_run()
    mlflow.log_param("mlflow_tracking_uri", mlflow_tracking_uri)

    params = all_params["prepare"]

    data_path = sys.argv[1]
    files = glob(os.path.join(data_path, "*.csv.gz"))

    mlflow.log_param("mlflow_tracking_uri", mlflow_tracking_uri)
    logger.info("Loading CSV files...")
    dfs = [
        pd.read_csv(file) for file in tqdm(files, desc="Loading CSV files")
    ]
    df = pd.concat(dfs, ignore_index=True)

    df = oversampling(df, params["oversampling"]["method"], params["oversampling"]["params"])

    sampling_path = os.path.join(data_path, "..", "..", "sampled")
    os.makedirs(sampling_path, exist_ok=True)
    
    logger.info("Saving oversampled CSV files...")
    files = save_split_csv(
        df=df,
        output_dir=sampling_path,
        output_prefix="cleaned",
        chunk_size=FIXED_ROWS
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

    mlflow.log_artifact("logs/sampling.log")
    mlflow.end_run()


if __name__ == "__main__":
    main()