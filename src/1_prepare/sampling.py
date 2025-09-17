import os
from glob import glob
from tqdm import tqdm
import sys
import yaml
import imblearn.over_sampling as imblearn_os
import pandas as pd

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


def main():
    FIXED_ROWS = 5_000_000
    if len(sys.argv) != 2:
        print("Usage: python sampling.py <data_path>")
        sys.exit(1)
    params = yaml.safe_load(open("params.yaml"))["prepare"]

    data_path = sys.argv[1]
    files = glob(os.path.join(data_path, "*.csv.gz"))

    dfs = [
        pd.read_csv(file) for file in tqdm(files, desc="Loading CSV files")
    ]
    df = pd.concat(dfs, ignore_index=True)

    df = oversampling(df, params["oversampling"]["method"], params["oversampling"]["params"])

    sampling_path = os.path.join(data_path, "..", "oversampled")
    os.makedirs(sampling_path, exist_ok=True)
    
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


if __name__ == "__main__":
    main()