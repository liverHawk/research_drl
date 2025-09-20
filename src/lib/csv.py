import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def save_split_csv(df, output_dir, output_prefix, chunk_size=100_000):
    arrays = []

    for i, start in enumerate(range(0, len(df), chunk_size)):
        path = os.path.join(output_dir, f"{output_prefix}_part{i+1}.csv.gz")
        arrays.append(
            [
                df.iloc[start:start+chunk_size],
                path
            ]
        )
    return arrays


def save_csv(df, path):
    df.to_csv(
        path,
        index=False,
        float_format="%.3f",
        compression="gzip"
    )


def multiprocess_save_csv(dfs, paths):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(save_csv, df, path) for df, path in zip(dfs, paths)
        ]
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                    pbar.update(1)
                except Exception as e:
                    print(f"Error during saving CSV files: {e}")