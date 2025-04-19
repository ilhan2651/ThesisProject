# app/services/stt_service.py

import os
import numpy as np
import soundfile as sf
import speech_recognition as sr
import torch
from pathlib import Path
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification
)

#  **GPU Kullanımı**
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {DEVICE}")

# 🎭 **Duygu Analizi Modeli** (yükle + eval)
_MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(_MODEL_NAME)
_emotion_model = (Wav2Vec2ForSequenceClassification
                  .from_pretrained(_MODEL_NAME, ignore_mismatched_sizes=True)
                  .to(DEVICE))
_emotion_model.eval()
_EMO_LABELS = _emotion_model.config.id2label

def _format_text(text: str) -> str:
    text = text.strip()
    for a, b in [(" .", "."), (" ,", ","), (" !", "!"), (" ?", "?"),
                 (" i ", " I "), (" im ", " I'm "), (" dont ", " don't "),
                 (" cant ", " can't ")]:
        text = text.replace(a, b)
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".!?":
        text += "."
    return text

def transcribe_audio(path: str) -> str | None:
    """Google Speech API ile metne çevirir."""
    r = sr.Recognizer()
    r.dynamic_energy_threshold = True
    with sr.AudioFile(path) as src:
        r.adjust_for_ambient_noise(src, duration=0.5)
        audio = r.record(src)
    try:
        txt = r.recognize_google(audio, language="en-US")
        return _format_text(txt)
    except (sr.UnknownValueError, sr.RequestError):
        return None

def predict_emotion(path: str) -> str | None:
    """Wav2Vec2 modeline sokup en olası duyguyu döner."""
    data, sr_rate = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr_rate != 16000:
        new_len = int(len(data) * 16000 / sr_rate)
        data = np.interp(
            np.linspace(0, len(data), new_len),
            np.arange(len(data)), data
        )
        sr_rate = 16000

    inputs = _feature_extractor(data, sampling_rate=sr_rate,
                                return_tensors="pt", padding=True)
    inputs = inputs.to(DEVICE)
    with torch.no_grad():
        logits = _emotion_model(inputs.input_values).logits
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().numpy()
    idx = int(np.argmax(probs))
    return _EMO_LABELS[idx]

def process_audio_file(path: str) -> dict:
    """
    1) Dosyayı transcribe eder.
    2) Ardından duygu tahmini yapar.
    3) {"text": ..., "emotion": ...} dict’i döner.
    """
    text = transcribe_audio(path)
    emotion = predict_emotion(path)
    return {"text": text, "emotion": emotion}
