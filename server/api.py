import subprocess

import pandas as pd
from fastapi import FastAPI

app = FastAPI()
MODELS_REGISTRY_PATH = "models.csv"
models_registry_df = pd.read_csv(
    MODELS_REGISTRY_PATH,
    sep=",",
    index_col="model"
)

@app.get("/train/{model_id}")
def train(model_id: str):
    docker_compose_path = str(models_registry_df.loc[
        model_id
    ]["train_entrypath"])
    run_compose = subprocess.run(
        [
            "docker", "compose", "-f", docker_compose_path, "run", "--build", "--rm", "train"
        ], capture_output=True, text=True, check=False
    )
    if run_compose.returncode != 0:
        return run_compose.stderr
    return run_compose.stdout
