import random
from pathlib import Path
from string import Template

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from tqdm import tqdm

RANDOM_STATE = 42
TEST_SIZE = 0.2
EPOCHS = 15_000
LEARNING_RATE = 1e-4
SR = 16_000

AUDIO_DATASET_PATH = Path("data/MELD_audio/archive/dev")
METADATA_PATH = Path("data/MELD_audio/dev.csv")
EMBEDDINGS_PATH = Path("data/embeddings/librosa.pt")
MODEL_PATH = Path("outputs/multitask.pt")
REPORT_PATH = Path("outputs/report.md")
REPORT_TEMPLATE_PATH = Path("lab2_report_template.md")

FRIENDS_NAMES = ["Phoebe", "Monica", "Ross", "Chandler", "Joey", "Rachel"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Не найден файл с разметкой: {METADATA_PATH}")

    metadata = pd.read_csv(METADATA_PATH)
    required_columns = {
        "Speaker",
        "Sentiment",
        "Dialogue_ID",
        "Utterance_ID",
    }
    if not required_columns.issubset(metadata.columns):
        raise ValueError(f"В {METADATA_PATH} не хватает нужных столбцов")

    metadata = metadata[metadata["Speaker"].isin(FRIENDS_NAMES)].copy()
    metadata["audio_path"] = metadata.apply(
        lambda row: AUDIO_DATASET_PATH
        / f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.flac",
        axis=1,
    )
    metadata = metadata[metadata["audio_path"].map(Path.exists)].reset_index(drop=True)

    if metadata.empty:
        raise RuntimeError("Не найдено ни одной записи с аудиофайлом")

    return metadata


def extract_librosa_features(audio_path, chunk_audio=True):
    def get_features(audio):
        mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=SR)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=SR)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=SR)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SR)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        rms = librosa.feature.rms(y=audio)

        features = np.concatenate(
            [
                mfcc.mean(axis=1),
                mfcc.std(axis=1),
                chroma.mean(axis=1),
                chroma.std(axis=1),
                spectral_centroid.mean(axis=1),
                spectral_centroid.std(axis=1),
                spectral_bandwidth.mean(axis=1),
                spectral_bandwidth.std(axis=1),
                spectral_rolloff.mean(axis=1),
                spectral_rolloff.std(axis=1),
                zero_crossing_rate.mean(axis=1),
                zero_crossing_rate.std(axis=1),
                rms.mean(axis=1),
                rms.std(axis=1),
            ]
        )
        return torch.from_numpy(features).float()

    audio, _ = librosa.load(audio_path, sr=SR, mono=True)
    if not chunk_audio:
        return get_features(audio)

    window_length = int(SR * 0.5)
    features = []
    for index in range(0, len(audio), window_length):
        window = audio[index : index + window_length]
        features.append(get_features(window))
    return torch.stack(features).mean(dim=0)


def load_embeddings(metadata):
    if EMBEDDINGS_PATH.exists():
        embeddings = torch.load(
            EMBEDDINGS_PATH,
            map_location="cpu",
            weights_only=True,
        )
        if len(embeddings) == len(metadata):
            return embeddings

    # Кеш нужен, чтобы не извлекать признаки при каждом запуске
    embeddings = []
    for audio_path in tqdm(metadata["audio_path"], desc="Librosa embeddings"):
        embeddings.append(extract_librosa_features(audio_path))

    embeddings = torch.stack(embeddings)
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, EMBEDDINGS_PATH)
    return embeddings


class MultiTask(nn.Module):
    def __init__(self, embedding_length, sentiment_count):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_length, 256),
            nn.ReLU(),
        )
        self.speaker_head = nn.Linear(256, len(FRIENDS_NAMES))
        self.sentiment_head = nn.Linear(256, sentiment_count)

    def forward(self, embeddings):
        features = self.encoder(embeddings)
        return self.speaker_head(features), self.sentiment_head(features)


def make_report(values):
    template = Template(REPORT_TEMPLATE_PATH.read_text(encoding="utf-8"))
    report = template.substitute(values)
    REPORT_PATH.write_text(report, encoding="utf-8")


def main():
    set_seed(RANDOM_STATE)
    device = get_device()
    metadata = load_metadata()
    embeddings = load_embeddings(metadata)
    sentiments = list(metadata["Sentiment"].unique())

    speaker_to_id = {name: index for index, name in enumerate(FRIENDS_NAMES)}
    sentiment_to_id = {
        sentiment: index for index, sentiment in enumerate(sentiments)
    }
    speaker_labels = torch.tensor(
        [speaker_to_id[name] for name in metadata["Speaker"]],
        dtype=torch.long,
    )
    sentiment_labels = torch.tensor(
        [sentiment_to_id[value] for value in metadata["Sentiment"]],
        dtype=torch.long,
    )

    (
        x_train,
        x_test,
        speaker_train,
        speaker_test,
        sentiment_train,
        sentiment_test,
    ) = train_test_split(
        embeddings,
        speaker_labels,
        sentiment_labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    x_train = torch.tensor(
        scaler.fit_transform(x_train),
        dtype=torch.float32,
        device=device,
    )
    x_test = torch.tensor(
        scaler.transform(x_test),
        dtype=torch.float32,
        device=device,
    )
    speaker_train = speaker_train.to(device)
    speaker_test = speaker_test.to(device)
    sentiment_train = sentiment_train.to(device)
    sentiment_test = sentiment_test.to(device)

    model = MultiTask(x_train.shape[1], len(sentiments)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    final_loss = 0.0
    for _ in tqdm(range(EPOCHS), desc="Training"):
        optimizer.zero_grad()
        speaker_logits, sentiment_logits = model(x_train)
        speaker_loss = criterion(speaker_logits, speaker_train)
        sentiment_loss = criterion(sentiment_logits, sentiment_train)
        loss = speaker_loss + sentiment_loss
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    model.eval()
    with torch.inference_mode():
        speaker_logits, sentiment_logits = model(x_test)
        speaker_predictions = torch.argmax(speaker_logits, dim=1)
        sentiment_predictions = torch.argmax(sentiment_logits, dim=1)

    speaker_report = classification_report(
        speaker_test.cpu(),
        speaker_predictions.cpu(),
        labels=list(range(len(FRIENDS_NAMES))),
        target_names=FRIENDS_NAMES,
        zero_division=0,
    )
    sentiment_report = classification_report(
        sentiment_test.cpu(),
        sentiment_predictions.cpu(),
        labels=list(range(len(sentiments))),
        target_names=sentiments,
        zero_division=0,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    report_values = {
        "samples_count": len(metadata),
        "train_count": len(x_train),
        "test_count": len(x_test),
        "speakers": ", ".join(FRIENDS_NAMES),
        "sentiments": ", ".join(sentiments),
        "embedding_size": embeddings.shape[1],
        "epochs": EPOCHS,
        "optimizer": "Adam",
        "lr": LEARNING_RATE,
        "loss_function": "CrossEntropyLoss",
        "final_loss": f"{final_loss:.6f}",
        "device": device,
        "speaker_report": speaker_report,
        "sentiment_report": sentiment_report,
        "model_path": MODEL_PATH,
    }
    make_report(report_values)


if __name__ == "__main__":
    main()
