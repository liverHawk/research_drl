import pandas as pd
from glob import glob

dir_path = "data/processed/normalized/train"
files = glob(f"{dir_path}/*.csv.gz")

dfs = [pd.read_csv(file) for file in files]
df = pd.concat(dfs, ignore_index=True)

print(df.iloc[0])