import os
import re

import gradio as gr
import librosa
import torch
from dotenv import load_dotenv
from torch import nn
from transformers import (
    BertModel,
    BertTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)

load_dotenv()

SR = int(os.getenv("SR", "16000"))
N_GENRES = int(os.getenv("N_GENRES", "6"))
MODEL_PATH = os.getenv("MODEL_PATH")

if MODEL_PATH is None:
    raise ValueError("Добавить путь к модели")

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

# Вспомогательные функции
def cut_audio(audio, max_len_sec=30):
    max_len_samples = max_len_sec * SR
    return audio[:max_len_samples]

# Эмбеддеры
wav2vec2_model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h"
)
wav2vec2_model = wav2vec2_model.eval()
wav2vec2_model = wav2vec2_model.to(device) # type: ignore
wav2vec2_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model = bert_model.eval()
bert_model = bert_model.to(device) # type: ignore


# Классификатор
class IntermediateFusion(nn.Module):
    def __init__(
            self, 
            n_genres,
            audio_dim,
            text_dim
        ):
        super().__init__()

        self.audio_proj = nn.Linear(audio_dim, 256)
        self.text_proj = nn.Linear(text_dim, 256)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            batch_first=True,
        )

        self.fusion = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),

            nn.Linear(256, 128),
            nn.GELU(),

            nn.Dropout(0.3),

            nn.Linear(128, n_genres)
        )

    def forward(self, audio_emb, text_emb):
        audio_emb = self.audio_proj(audio_emb)
        text_emb = self.text_proj(text_emb)

        x = torch.stack(
            [audio_emb, text_emb],
            dim=1
        )

        x = self.fusion(x)

        x = x.mean(dim=1)

        logits = self.classifier(x)

        return logits

model = IntermediateFusion(N_GENRES, 768, 768)
state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# Перевод на человеческий
nums_to_genres = {
    0: 'R&B/Soul',
    1: 'Rock/Alternative',
    2: 'Pop',
    3: 'Hip-Hop/Rap',
    4: 'Country',
    5: 'Dance/Electronic'
}
    
def process(audio, text):
    def get_audio_emb(audio):
        audio, _ = librosa.load(audio, sr=SR)
        audio = cut_audio(audio)

        inputs = wav2vec2_feature_extractor(
            audio,
            sampling_rate=SR,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.inference_mode():
            outputs = wav2vec2_model(**inputs)

        # [1, time, 768] -> [1, 768]
        return outputs.last_hidden_state.mean(dim=1)

    def get_text_emb(text):
        lines = [
          re.sub(r"\[.+\]", "", line.strip())
          for line in text.splitlines()
        ]
        lines = [line for line in lines if line]

        if not lines:
            lines = [""]
        inputs = bert_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.inference_mode():
            outputs = bert_model(**inputs).last_hidden_state

        # [1, tokens, 768] -> [1, 768]
        return outputs.mean(dim=0).mean(dim=0).unsqueeze(0)

    audio_emb = get_audio_emb(audio)
    text_emb = get_text_emb(text)

    with torch.inference_mode():
        logits = model(audio_emb, text_emb)
    probabilities = torch.softmax(logits, dim=1)
    for genre_id, probability in enumerate(probabilities[0]):
        genre = nums_to_genres[genre_id]
        print(f"{genre}: {probability.item():.2%}")

    preds = int(torch.argmax(logits, dim=1).item())

    return nums_to_genres[preds]

app = gr.Interface(
    fn=process,
    inputs=[
        gr.Audio(type="filepath", label="Аудио"),
        gr.Textbox(label="Текст"),
    ],
    outputs=[
        gr.Textbox(label="Текст"),
    ],
)


app.launch(server_name="0.0.0.0", server_port=5000)
