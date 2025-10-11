import os
import mlflow
import numpy as np
import matplotlib.pyplot as plt

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# credential = DefaultAzureCredential()
# credential.get_token("https://management.azure.com/.default")

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config.json"))
print(path)

ml_client = MLClient.from_config(
    credential=DefaultAzureCredential(),
    path=path
)

tracking_uri = ml_client.workspaces.get(
    ml_client.workspace_name
).mlflow_tracking_uri

mlflow.set_tracking_uri(tracking_uri)

with mlflow.start_run():
    mlflow.log_param("param1", "value1")
    mlflow.log_params(
        {
            "param2": "value2",
            "param3": 3,
            "param4": 4.0
        }
    )
    mlflow.log_metric("metric1", 0.95)
    mlflow.log_metrics(
        {
            "metric2": 0.85,
            "metric3": 0.75
        }
    )
    with open("output.txt", "w") as f:
        f.write("This is a test artifact.")

    mlflow.log_artifact("output.txt")

    array = [1, 2, 3, 4, 5]
    os.makedirs("output_dir", exist_ok=True)
    for i in array:
        with open(f"output_dir/array_{i}.txt", "w") as f:
            f.write("\n".join(map(str, array)))
    mlflow.log_artifacts("output_dir")

    dictionary = {"key1": "value1", "key2": "value2"}
    mlflow.log_dict(dictionary, "dictionary.json")

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    mlflow.log_figure(fig, "plot.png")
    plt.close(fig)

    input_array = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    dataset = mlflow.data.from_numpy(input_array, source="input_array.csv")
    mlflow.log_input(dataset, context="input_data")
