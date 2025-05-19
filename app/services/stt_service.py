import os
import numpy as np
import soundfile as sf
import torch
from transformers import WavLMForSequenceClassification, Wav2Vec2FeatureExtractor, pipeline
from pydub import AudioSegment
from speech_recognition import Recognizer, AudioFile, AudioData
from deepmultilingualpunctuation import PunctuationModel
import re

# Cihaz belirleme
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Model yolları
MODEL_PATH = "models/emotion_model"
_emotion_model = WavLMForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
_emotion_model.eval()
_text_emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
_punctuation_model = PunctuationModel()

_EMO_LABELS = {0: "anger", 1: "happy", 2: "neutral", 3: "sad", 4: "surprise"}

def transcribe_audio(path):
    recognizer = Recognizer()
    if path.endswith(".mp3"):
        sound = AudioSegment.from_mp3(path)
        path = path.replace(".mp3", ".wav")
        sound.export(path, format="wav")

    with AudioFile(path) as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio, language="en-US")
        except:
            return ""

def predict_emotion_from_audio(path):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        new_len = int(len(data) * 16000 / sr)
        data = np.interp(np.linspace(0, len(data), new_len), np.arange(len(data)), data)
        sr = 16000

    inputs = _feature_extractor(data, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _emotion_model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        conf = probs[0][pred].item()
    return {"emotion": _EMO_LABELS[pred], "confidence": conf}

def process_audio_file(path):
    raw_text = transcribe_audio(path)
    if not raw_text:
        return {"text": "", "emotion": "neutral"}

    punctuated = _punctuation_model.restore_punctuation(raw_text)
    sentences = [s.strip() for s in re.split(r'[.!?]', punctuated) if s.strip()]
    sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
    original_punctuation = re.findall(r'[.!?]', punctuated)

    audio_data, sr = sf.read(path)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    frame_len = int(0.1 * sr)
    energy = np.array([np.sum(np.abs(audio_data[i:i+frame_len])) for i in range(0, len(audio_data), frame_len)])
    threshold = np.mean(energy) * 0.5
    silent_regions = np.where(energy < threshold)[0]

    segments, start_idx = [], 0
    for i in range(1, len(silent_regions)):
        if silent_regions[i] - silent_regions[i-1] > 5:
            end_idx = silent_regions[i-1] * frame_len
            if end_idx - start_idx > sr:
                segments.append(audio_data[start_idx:end_idx])
            start_idx = end_idx
    if len(audio_data) - start_idx > sr:
        segments.append(audio_data[start_idx:])

    audio_emotions, text_emotions = [], []
    for sentence in sentences:
        results = _text_emotion_analyzer(sentence)
        max_emo = max(results[0], key=lambda x: x['score'])
        text_emotions.append({"emotion": max_emo['label'], "confidence": max_emo['score']})

    for i, segment in enumerate(segments):
        temp_path = f"temp_segment_{i}.wav"
        sf.write(temp_path, segment, sr)
        emo = predict_emotion_from_audio(temp_path)
        if emo: audio_emotions.append(emo)
        os.remove(temp_path)

    final_emotions = []
    for i in range(min(len(text_emotions), len(audio_emotions))):
        t, a = text_emotions[i], audio_emotions[i]
        if t['emotion'] in ['fear', 'disgust'] or (t['confidence'] >= 0.8 and t['emotion'] != 'neutral'):
            final_emotions.append(t)
        elif t['emotion'] == 'neutral' and a['confidence'] >= 0.7:
            final_emotions.append(a)
        elif a['confidence'] >= 0.7:
            final_emotions.append(a)
        else:
            final_emotions.append(t)

    # Duygu-emoji eşleştirmesi
    emotion_emojis = {
        "anger": "😠",
        "happy": "😄",
        "neutral": "😐",
        "sad": "😢",
        "sadness": "😢",
        "surprise": "😲",
        "fear": "😨",
        "disgust": "🤢"
    }

    text_with_emotions = ""
    current_emotion = None
    buffer = []

    for i, (sent, emo) in enumerate(zip(sentences, final_emotions)):
        punctuation = original_punctuation[i] if i < len(original_punctuation) else "."
        emotion = emo['emotion']
        emoji = emotion_emojis.get(emotion, "😐")  # Eğer duygu bulunamazsa nötr emoji kullan

        if emotion == current_emotion:
            buffer.append(sent.strip() + punctuation)
        else:
            if buffer:
                buffer[-1] = buffer[-1].rstrip(punctuation) + f" {emotion_emojis.get(current_emotion, '😐')}" + punctuation
                text_with_emotions += " ".join(buffer) + " "
            buffer = [sent.strip() + punctuation]
            current_emotion = emotion

    if buffer:
        buffer[-1] = buffer[-1].rstrip(punctuation) + f" {emotion_emojis.get(current_emotion, '😐')}" + punctuation
        text_with_emotions += " ".join(buffer)

    text_with_emotions = text_with_emotions.strip()
    overall = _text_emotion_analyzer(punctuated)
    overall_emo = max(overall[0], key=lambda x: x['score'])

    return {"text": text_with_emotions, "emotion": overall_emo['label']}