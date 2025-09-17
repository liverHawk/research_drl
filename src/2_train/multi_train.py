import yaml
import pandas as pd
import os
from glob import glob


def load_csv():
    path = os.path.join("data", "processed", "sampled")
    files = glob(os.path.join(path, "*.csv.gz"))
    df = pd.concat([
        pd.read_csv(f) for f in files
    ])
    return df


def train(df, params):
    env = 


def main():
    params = yaml.safe_load(open("params.yaml"))["train"]

    df = load_csv()

    train(df, params)


if __name__ == "__main__":
    main()