import speech_recognition as sr
from deepmultilingualpunctuation import PunctuationModel
import warnings
import torch

# ✅ Gereksiz uyarıları kapat
warnings.simplefilter("ignore", category=UserWarning)

# ✅ GPU Kullanımı
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# ✅ Noktalama modelini yükle ve GPU'ya taşı
punctuation_model = PunctuationModel("kredor/punctuate-all")
punctuation_model.pipe.model.to(device)  # ✅ DOĞRU YER BURASI!

def recognize_speech():
    recognizer = sr.Recognizer()
    
    # 🔹 Algılama hassasiyetini optimize et
    recognizer.energy_threshold = 300  # Daha düşük sesleri filtrele
    recognizer.dynamic_energy_threshold = True  # Otomatik ses seviyesi ayarı
    
    with sr.Microphone(sample_rate=44100) as source:  # ✅ Örnekleme frekansı 44100 Hz
        print("🎤 Konuşmaya başlayın (Kaydı durdurmak için 2.5 saniye sessiz kalın)...")
        
        recognizer.adjust_for_ambient_noise(source, duration=2)  # Gürültü ayarını 2 saniyeye çıkar
        
        recognizer.pause_threshold = 2.5  # Sessizlik süresi arttırıldı (Nefes alırken kesilmesin)
        audio = recognizer.listen(source)

    try:
        print("📝 Metne çeviriliyor...")
        text = recognizer.recognize_google(audio, language="tr-TR")  # ✅ Google’ın ücretsiz API’si kullanılıyor

        # ✅ Kelimeler arasına boşlukları ekleyelim
        formatted_text = " ".join(text.split())  # ✅ Kelimeleri ayır

        # ✅ Noktalama ekleyelim (GPU'da çalışacak şekilde)
        with torch.no_grad():  # GPU kullanımını optimize etmek için
            punctuated_text = punctuation_model.restore_punctuation(formatted_text)

        print("📄 Sonuç:", punctuated_text)  # 🔹 SADECE NOKTALAMA EKLENMİŞ METİNİ YAZDIR

        return punctuated_text
    except sr.UnknownValueError:
        print("❌ Google Speech Recognition sesi anlayamadı.")
        return None
    except sr.RequestError:
        print("❌ Google API'ye bağlanılamadı. İnternet bağlantınızı kontrol edin.")
        return None

if __name__ == "__main__":
    recognize_speech()
