import json
import os
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import torch
from dotenv import load_dotenv
from torch import nn

load_dotenv()

SR = 16000

MODEL_NAME = Path(os.getenv("MODEL_NAME", "multitask.pt"))
CONFIG_NAME = Path(os.getenv("CONFIG_NAME", "config.json"))
MODEL_PATH = Path(__file__).resolve().parent / MODEL_NAME
CONFIG_PATH = Path(__file__).resolve().parent / CONFIG_NAME

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
FRIENDS_NAMES = config["friends_names"]
SENTIMENT_CATEGORIES = config["setiment_categories"]

class MultiTask(nn.Module):
    def __init__(self, emb_length):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(emb_length, 256),
            nn.ReLU(),
        )
        self.speaker_head = nn.Sequential(
            nn.Linear(256, len(FRIENDS_NAMES)),
        )
        self.sentiment_head = nn.Sequential(
            nn.Linear(256, len(SENTIMENT_CATEGORIES)),
        )

    def forward(self, emb):
        features = self.encoder(emb)
        return self.speaker_head(features), self.sentiment_head(features)


state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
emb_length = state_dict["encoder.0.weight"].shape[1]

model = MultiTask(emb_length)
model.load_state_dict(state_dict)
model.eval()


def extract_features(audio_path):
    def get_features(audio):
        mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=SR)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=SR)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=SR)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SR)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        rms = librosa.feature.rms(y=audio)

        return np.concatenate([
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
        ])

    audio, _ = librosa.load(audio_path, sr=SR, mono=True)
    if len(audio) == 0:
        raise gr.Error("Запись пустая")

    window_length = int(SR * 0.5)
    windows = []

    for start in range(0, len(audio), window_length):
        window = audio[start : start + window_length]
        windows.append(get_features(window))

    features = np.mean(windows, axis=0)
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)


def predict(audio_path):
    if not audio_path:
        raise gr.Error("Сначала запишите или загрузите аудио")

    features = extract_features(audio_path)

    with torch.inference_mode():
        speaker_logits, sentiment_logits = model(features)

    speaker_id = int(torch.argmax(speaker_logits, dim=1).item())
    sentiment_id = int(torch.argmax(sentiment_logits, dim=1).item())

    return FRIENDS_NAMES[speaker_id], SENTIMENT_CATEGORIES[sentiment_id]


with gr.Blocks(title="Friends Voice") as app:
    audio = gr.Audio(
        sources=["microphone", "upload"],
        type="filepath",
        label="Аудио",
    )
    button = gr.Button("Определить")

    speaker = gr.Textbox(label="Герой FRIENDS")
    sentiment = gr.Textbox(label="Сентимент")

    button.click(predict, inputs=audio, outputs=[speaker, sentiment])
    audio.stop_recording(predict, inputs=audio, outputs=[speaker, sentiment])
    audio.upload(predict, inputs=audio, outputs=[speaker, sentiment])


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=5000)
