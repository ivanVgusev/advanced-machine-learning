import pandas as pd
import subprocess


MODELS_REGISTRY_PATH = "models.csv"
models_registry_df = pd.read_csv(
    MODELS_REGISTRY_PATH,
    sep=",",
    index_col="model"
)

docker_compose_path = str(models_registry_df.loc[
    "lab1"
]["train_entrypath"])
run_compose = subprocess.run(
    [
        "docker", "compose", "-f", docker_compose_path, "run", "--build", "--rm", "train"
    ], capture_output=True, text=True
)
print("Return code:", run_compose.returncode)
print("STDOUT:", run_compose.stdout)
print("STDERR:", run_compose.stderr)
