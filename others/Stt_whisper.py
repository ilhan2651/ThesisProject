import torch
import whisper
import sounddevice as sd
import numpy as np
import wave
import time
import threading
import keyboard  # Klavyeden giriş almak için

# 🚀 GPU Kullanım Kontrolü
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# 🎙️ Whisper Modelini Yükle
model = whisper.load_model("medium", device=device)

# 🎤 Mikrofon Konfigürasyonu
fs = 44100  # Örnekleme frekansı
channels = 1  # Tek kanal (Mono)
chunk_size = 1024  # Ses örnekleme boyutu
recording = True  # Kayıt durumunu kontrol eden değişken
mic_id = 1

def record_audio():
    global recording
    print("🎤 Konuşmaya başlayabilirsiniz... (Kaydı durdurmak için 'q' tuşuna basın)")

    buffer = []
    
    def callback(indata, frames, time, status):
        if not recording:
            raise sd.CallbackStop  # Eğer kayıt durdurulmuşsa, kaydı kes

        buffer.append(indata.copy())

    def stop_listener():
        global recording
        keyboard.wait("q")  # Kullanıcı 'q' tuşuna basana kadar bekle
        recording = False
        print("\n🛑 'q' tuşuna basıldı, kayıt durduruluyor...")

    threading.Thread(target=stop_listener, daemon=True).start()

    try:
        with sd.InputStream(device=mic_id,samplerate=fs, channels=channels, callback=callback, blocksize=chunk_size):
            while recording:
                sd.sleep(100)
    except sd.CallbackStop:
        pass

    # Kaydedilen sesi tek bir numpy dizisine birleştir
    audio_data = np.concatenate(buffer, axis=0)

    # WAV dosyası olarak kaydet
    filename = "realtime_audio.wav"
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_data.astype(np.int16).tobytes())

    print(f"🎧 Kayıt tamamlandı. Dosya kaydedildi: {filename}")
    return filename

def transcribe_audio(filename):
    print("📝 Metne çeviriliyor...")
    result = model.transcribe(filename, fp16=torch.cuda.is_available())  # CUDA destekliyorsa fp16 modunu aç
    print("📄 Metin Çıktısı:", result["text"])

def record_and_transcribe():
    filename = record_audio()
    transcribe_audio(filename)

if __name__ == "__main__":
    record_and_transcribe()
