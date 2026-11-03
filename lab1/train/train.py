import random
import re
from collections import OrderedDict
from pathlib import Path
from string import Template

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm
from transformers import (
    BertModel,
    BertTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)

SR = 16_000
RANDOM_STATE = 42
TEST_SIZE = 0.2
EPOCHS = 150
LEARNING_RATE = 1e-4

# Пути к аудио, текстам песен и жанровой разметке
AUDIO_CORPUS_PATH = Path("data/PMEmo2019/chorus")
LYRICS_CORPUS_PATH = Path("data/PMEmo2019/lyrics")
ANNOTATIONS_PATH = Path("data/PMEmo2019/genre_annotations/genre_annotations.csv")
AUDIO_EMBEDDINGS_PATH = Path("data/embeddings/wav2vec2.pt")
TEXT_EMBEDDINGS_PATH = Path("data/embeddings/bert.pt")
REPORT_PATH = Path("outputs/report.md")
REPORT_TEMPLATE_PATH = Path("lab1_report_template.md")
if not REPORT_TEMPLATE_PATH.exists():
    raise FileNotFoundError("Не найден файл с темплейтом отчета")

WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base-960h"
BERT_MODEL_NAME = "bert-base-uncased"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    # Фиксируем сиды
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_lrc(text_path):
    def lrc_preprocess(line):
        # Убираем временные метки и пустые края строки
        line = line.strip()
        return re.sub(r"\[.+?\]", "", line).strip()

    with text_path.open("r", encoding="utf-8", errors="ignore") as file:
        lines = [lrc_preprocess(line) for line in file]

    lines = [line for line in lines if line]
    return lines or [""]


def load_dataset_paths():
    # Сначала проверяем, что разметка существует и имеет нужный формат
    if not ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Не найден файл с жанрами: {ANNOTATIONS_PATH}."
        )

    annotations_df = pd.read_csv(ANNOTATIONS_PATH)
    required_columns = {"musicId", "genre"}
    if not required_columns.issubset(annotations_df.columns):
        raise ValueError(
            f"В {ANNOTATIONS_PATH} должны быть столбцы musicId и genre."
        )

    annotations_df = annotations_df.set_index("musicId")
    samples = []

    # Собираем только полные тройки: аудио, текст и жанр
    for audio_path in sorted(AUDIO_CORPUS_PATH.glob("*.mp3"), key=lambda path: int(path.stem)):
        music_id = int(audio_path.stem)
        text_path = LYRICS_CORPUS_PATH / f"{music_id}.lrc"

        if music_id not in annotations_df.index or not text_path.exists():
            continue

        genre = annotations_df.at[music_id, "genre"]
        if isinstance(genre, pd.Series):
            raise ValueError(f"musicId={music_id} встречается в annotations несколько раз.")

        samples.append((audio_path, text_path, str(genre)))

    if not samples:
        raise RuntimeError("Не найдено ни одной полной пары audio + lyrics + genre.")

    return samples


def extract_audio_embeddings(audio_paths, device):
    # Загружаем Wav2Vec2 один раз для всей коллекции аудио
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        WAV2VEC2_MODEL_NAME
    )
    model = Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL_NAME).to(device)
    model.eval()

    embeddings = []
    for audio_path in tqdm(audio_paths, desc="Wav2Vec2 embeddings"):
        audio, _ = librosa.load(audio_path, sr=SR)
        inputs = feature_extractor(
            audio,
            sampling_rate=SR,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.inference_mode():
            features = model(**inputs).last_hidden_state

        # Усредняем признаки по времени до одного вектора на трек
        embeddings.append(features.mean(dim=1).squeeze(0).cpu())

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return torch.stack(embeddings)


def extract_text_embeddings(text_paths, device):
    # Загружаем BERT и обрабатываем текст построчно
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertModel.from_pretrained(BERT_MODEL_NAME).to(device)
    model.eval()

    embeddings = []
    for text_path in tqdm(text_paths, desc="BERT embeddings"):
        lyrics_lines = read_lrc(text_path)
        inputs = tokenizer(
            lyrics_lines,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.inference_mode():
            features = model(**inputs).last_hidden_state

        # Получаем один общий вектор для всех строк песни
        embeddings.append(features.mean(dim=0).mean(dim=0).cpu())

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return torch.stack(embeddings)


def load_embeddings(paths, embeddings_path, extract_embeddings, device):
    if embeddings_path.exists():
        embeddings = torch.load(
            embeddings_path,
            map_location="cpu",
            weights_only=True,
        )
        if len(embeddings) == len(paths):
            return embeddings

    # Кеш нужен, чтобы не извлекать признаки при каждом запуске
    embeddings = extract_embeddings(paths, device)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, embeddings_path)
    return embeddings


def make_report(sub_dict: dict) -> None:
    report_template = Template(REPORT_TEMPLATE_PATH.read_text())
    report = report_template.safe_substitute(sub_dict)
    REPORT_PATH.write_text(str(report))


class IntermediateFusion(nn.Module):
    def __init__(self, n_genres, audio_dim, text_dim):
        super().__init__()

        # Приводим аудио и текст к одинаковой размерности
        self.audio_proj = nn.Linear(audio_dim, 256)
        self.text_proj = nn.Linear(text_dim, 256)

        # Трансформер учится учитывать обе модальности вместе
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            batch_first=True,
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # По объединенному вектору предсказываем жанр
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_genres),
        )

    def forward(self, audio_emb, text_emb):
        audio_emb = self.audio_proj(audio_emb)
        text_emb = self.text_proj(text_emb)
        # Складываем две модальности в короткую последовательность
        fused = torch.stack([audio_emb, text_emb], dim=1)
        fused = self.fusion(fused).mean(dim=1)
        return self.classifier(fused)


def main():
    # Готовим окружение
    set_seed(RANDOM_STATE)
    device = get_device()
    print(f"Device: {device}")

    samples = load_dataset_paths()
    audio_paths = [sample[0] for sample in samples]
    text_paths = [sample[1] for sample in samples]
    annotations = [sample[2] for sample in samples]

    # Назначаем каждому названию жанра числовую метку
    genres = list(OrderedDict.fromkeys(annotations))
    genre_to_num = {genre: index for index, genre in enumerate(genres)}
    labels = torch.tensor(
        [genre_to_num[genre] for genre in annotations],
        dtype=torch.long,
    )

    print(f"Samples: {len(samples)}")
    print(f"Genres: {genres}")

    audio_embeddings = load_embeddings(
        audio_paths,
        AUDIO_EMBEDDINGS_PATH,
        extract_audio_embeddings,
        device,
    )
    text_embeddings = load_embeddings(
        text_paths,
        TEXT_EMBEDDINGS_PATH,
        extract_text_embeddings,
        device,
    )

    # Трейн-тест сплит
    audio_train, audio_test, text_train, text_test, y_train, y_test = train_test_split(
        audio_embeddings,
        text_embeddings,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    audio_train = audio_train.to(device)
    text_train = text_train.to(device)
    y_train = y_train.to(device)
    audio_test = audio_test.to(device)
    text_test = text_test.to(device)

    # Создаем модель, функцию ошибки и оптимизатор
    model = IntermediateFusion(
        n_genres=len(genres),
        audio_dim=audio_embeddings.shape[1],
        text_dim=text_embeddings.shape[1],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Обучаем классификатор
    model.train()
    for _ in tqdm(range(EPOCHS), desc="Training"):
        optimizer.zero_grad()
        logits = model(audio_train, text_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    # Получаем предсказания без расчета градиентов
    model.eval()
    with torch.inference_mode():
        logits = model(audio_test, text_test)
        predictions = torch.argmax(logits, dim=1)

    results = classification_report(
        y_test.cpu().numpy(),
        predictions.cpu().numpy(),
        labels=list(range(len(genres))),
        target_names=genres,
        zero_division=0,
    )

    report_values = {
        "samples_count": len(samples),
        "genres": ", ".join(genres),
        "optimizer": "Adam",
        "lr": LEARNING_RATE,
        "loss_function": "CrossEntropyLoss",
        "device": device,
        "classification_report": results,
    }
    make_report(report_values)


if __name__ == "__main__":
    main()
