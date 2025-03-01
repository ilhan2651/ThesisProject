import torch
import soundfile as sf
import numpy as np
import sounddevice as sd
import time
import speech_recognition as sr
import keyboard
import librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

# 🚀 **GPU Kullanımı**
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# 🎭 **Duygu Analizi Modeli**
emotion_model_name = "r-f/wav2vec-english-speech-emotion-recognition"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(emotion_model_name)
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(emotion_model_name).to(device)

# 🎤 **Mikrofon ile Ses Kaydı (Sadece 'q' tuşu ile durur)**
def record_audio(filename="speech.wav"):
    fs = 16000
    channels = 1
    chunk_size = 1024
    buffer = []
    recording_active = True

    print("🎤 Konuşmaya başlayın ('q' tuşuna basarak kaydı durdurabilirsiniz)...")

    def callback(indata, frames, time_info, status):
        nonlocal recording_active
        if keyboard.is_pressed("q"):
            print("🛑 'q' tuşuna basıldı. Kayıt durduruluyor...")
            recording_active = False
            raise sd.CallbackStop
        buffer.append(indata.copy())

    with sd.InputStream(samplerate=fs, channels=channels, callback=callback, blocksize=chunk_size):
        while recording_active:
            time.sleep(0.1)

    audio_data = np.concatenate(buffer, axis=0)
    sf.write(filename, audio_data, fs)

    return filename

# 🎙️ **Google Speech API ile Metne Çevirme**
def transcribe_audio(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        print("📝 Metne çeviriliyor...")
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language="tr-TR")  
        text = text.capitalize()
        if not text.endswith((".", "?", "!")):
            text += "."
        print(f"📄 Metin Çıktısı: {text}")
        return text
    except sr.UnknownValueError:
        print("❌ Google Speech Recognition sesi anlayamadı.")
        return None
    except sr.RequestError:
        print("❌ Google API'ye bağlanılamadı.")
        return None

# 🎭 **Duygu Analizi (Güncellenmiş Algoritma)**
def predict_emotion(filename):
    print("🔍 Duygu analizi yapılıyor...")

    # **Ses dosyasını yükle**
    audio, sr = librosa.load(filename, sr=16000)

    # **Hugging Face modeline uygun formatta giriş verisi hazırla**
    inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = inputs.to(device)

    # ✅ **Model ile tahmin yap**
    with torch.no_grad():
        outputs = emotion_model(inputs.input_values)
        predictions = torch.nn.functional.softmax(outputs.logits.mean(dim=1), dim=-1)  # Ortalama alarak stabil tahmin yap

    # ✅ **En yüksek olasılığa sahip duyguyu bul**
    predicted_label = torch.argmax(predictions, dim=-1).item()
    emotion = emotion_model.config.id2label[predicted_label]

    # ✅ **Duygu yüzdelerini göster**
    emotion_scores = {emotion_model.config.id2label[i]: round(predictions[0][i].item() * 100, 2) for i in range(len(predictions[0]))}
    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)

    print("\n🎭 **Duygu Olasılıkları:**")
    for emo, score in sorted_emotions:
        print(f"   - {emo}: %{score}")

    return emotion

# 🎯 **Ses Kaydet + Metne Çevir + Duygu Analizi Yap**
if __name__ == "__main__":
    audio_file = record_audio()
    text = transcribe_audio(audio_file)  
    predict_emotion(audio_file)
