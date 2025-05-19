import torch
import soundfile as sf
import numpy as np
import sounddevice as sd
import time
import speech_recognition as sr
import keyboard
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import os
from pathlib import Path
import wave

#  **GPU Kullanımı**
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# 🎭 **Duygu Analizi Modeli**
emotion_model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(emotion_model_name)
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    emotion_model_name,
    ignore_mismatched_sizes=True
).to(device)

# Modeli değerlendirme moduna al
emotion_model.eval()

# Modelin kendi etiketlerini kullan
emotion_labels = emotion_model.config.id2label

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
    
    # Gürültü ayarları ve hassasiyet
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 300
    recognizer.pause_threshold = 0.8
    recognizer.phrase_threshold = 0.3
    recognizer.non_speaking_duration = 0.5

    with sr.AudioFile(filename) as source:
        print("📝 Metne çeviriliyor...")
        # Gürültü azaltma
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        # Sesi kaydet
        audio = recognizer.record(source)
    
    try:
        # İngilizce olarak tanıma yap
        text = recognizer.recognize_google(audio, language="en-US")
        
        # Metni düzenle
        text = format_text(text)
        print(f"📄 Metin Çıktısı: {text}")
        return text
        
    except sr.UnknownValueError:
        print("❌ Ses tanıma başarısız oldu.")
        return None
    except sr.RequestError as e:
        print(f"❌ API hatası: {str(e)}")
        return None
    except Exception as e:
        print(f"❌ Beklenmeyen hata: {str(e)}")
        return None

def format_text(text):
    """Metni düzenle ve noktalama işaretlerini ekle"""
    # Temel düzeltmeler
    text = text.strip()
    
    # Noktalama işaretlerini düzelt
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" !", "!")
    text = text.replace(" ?", "?")
    
    # Temel kelime düzeltmeleri
    text = text.replace(" i ", " I ")
    text = text.replace(" im ", " I'm ")
    text = text.replace(" ill ", " I'll ")
    text = text.replace(" dont ", " don't ")
    text = text.replace(" cant ", " can't ")
    
    # Cümle başını büyük harf yap
    if text:
        text = text[0].upper() + text[1:]
    
    # Cümle sonuna nokta ekle
    if text and not text.endswith(('.', '!', '?')):
        text += "."
    
    return text

# 🎭 **Duygu Analizi (Güncellenmiş Algoritma)**
def predict_emotion(filename):
    print("🔍 Duygu analizi yapılıyor...")
    try:
        audio, sr = sf.read(filename)
        
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        if sr != 16000:
            new_length = int(len(audio) * 16000 / sr)
            audio = np.interp(
                np.linspace(0, len(audio), new_length),
                np.arange(len(audio)),
                audio
            )
            sr = 16000

        inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = emotion_model(inputs.input_values)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            if len(predictions.shape) > 1:
                predictions = predictions.squeeze(0)
            
            predictions = predictions.cpu().numpy()

        predicted_label = np.argmax(predictions)
        emotion = emotion_labels[predicted_label]

        emotion_scores = {
            emotion_labels[i]: round(float(predictions[i]) * 100, 2)
            for i in range(len(predictions))
        }
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("\n🎭 Duygu Olasılıkları:")
        for emo, score in sorted_emotions:
            print(f"   - {emo}: %{score}")

        return emotion
        
    except Exception as e:
        print(f"Duygu analizi sırasında hata: {str(e)}")
        return None

def process_audio_file(audio_path):
    """Ses dosyasını işle ve duygu analizi yap"""
    try:
        print(f"Ses dosyası yükleniyor: {audio_path}")
        
        # MP3'ü WAV'a dönüştür
        if audio_path.lower().endswith('.mp3'):
            import pydub
            sound = pydub.AudioSegment.from_mp3(audio_path)
            wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
            sound.export(wav_path, format="wav")
            audio_path = wav_path
        
        # Ses dosyasını transcribe et
        text = transcribe_audio(audio_path)
        print(f"\nMetin: {text}")
        
        # Duygu analizi yap
        emotion = predict_emotion(audio_path)
        print(f"\nDuygu Analizi Sonuçları: {emotion}")
        
        return text, emotion
    except Exception as e:
        print(f"Hata: {str(e)}")
        return None, None

def process_audio_folder(folder_path):
    """Klasördeki tüm ses dosyalarını işle"""
    print("Ses dosyaları analiz ediliyor...")
    
    # Desteklenen ses formatları
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg'}
    
    # Klasördeki ses dosyalarını bul
    audio_files = [f for f in Path(folder_path).glob('*') if f.suffix.lower() in audio_extensions]
    
    if not audio_files:
        print("Klasörde ses dosyası bulunamadı!")
        return
    
    # Her ses dosyasını işle
    for audio_file in audio_files:
        print(f"\n{'-'*50}")
        print(f"Dosya işleniyor: {audio_file.name}")
        text, emotion = process_audio_file(str(audio_file))

# 🎯 **Ses Kaydet + Metne Çevir + Duygu Analizi Yap**
if __name__ == "__main__":
    # Ses dosyalarının bulunduğu klasör
    audio_folder = "ses_dosyalari"  # Bu klasörü projenizde oluşturun
    
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)
        print(f"'{audio_folder}' klasörü oluşturuldu. Lütfen ses dosyalarınızı buraya ekleyin.")
    else:
        process_audio_folder(audio_folder)
