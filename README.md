# 🧠 Emotion-Aware Speech Processing System (TTS & STT)

This project is part of a thesis study that integrates **Speech-to-Text (STT)** and **Text-to-Speech (TTS)** technologies with **emotion recognition**, enabling more expressive and accessible communication—especially for visually and hearing-impaired users.

---

## 🔧 Technologies & Tools

* **Python 3.10+**
* **Hugging Face Transformers** (`WavLM`, `Roberta`)
* **Amazon Polly (SSML-based)**
* **FastAPI** – RESTful API
* **SpeechRecognition** + `Google Web Speech API`
* **deepmultilingualpunctuation\`** – Sentence segmentation
* **Pydub / SoundFile / Numpy** – Audio handling

---

## 📁 Project Structure

```
project_root/
├── app/
│   ├── routes/
│   │   ├── tts_route.py         # /generate-audio endpoint
│   │   └── stt_route.py         # /stt/analyze endpoint
│   └── services/
│       ├── tts_helper.py        # Emotion-enhanced SSML synthesis
│       └── stt_service.py       # Transcribe & detect emotions
├── models/
│   └── request_models.py        # TTSRequest model
├── main.py                      # FastAPI app
├── models/emotion_model         # Fine-tuned WavLM model dir
└── ses_dosyalari/               # Uploaded/generated audio files
```

---

## 🗣️ Speech-to-Text (STT)

* Transcribes uploaded `.wav` or `.mp3` files to text
* Adds punctuation using a multilingual model
* Segments sentences and analyzes:

  * Text-based emotions (Roberta model)
  * Audio-based emotions (WavLM fine-tuned model)
* Returns **combined emotional analysis** and annotated sentences with emojis

**API Endpoint:**

```http
POST /stt/analyze
```

**Returns:**

```json
{
  "text": "He is excited 😄. I'm scared 😨.",
  "emotion": "fear"
}
```

---

## 🔊 Text-to-Speech (TTS)

* Takes a text input, analyzes each sentence for emotion
* Adjusts voice parameters dynamically (rate, pitch, volume)
* Applies **SSML tags** for emotional expression
* Uses **Amazon Polly** with `Joanna` voice

**API Endpoint:**

```http
POST /generate-audio
Body: {
  "text": "I'm so happy today! But yesterday was awful."
}
```

**Returns:**

* `.mp3` audio file synthesized with emotional variation

---

## 💡 Emotion Models Used

* **Text**: `j-hartmann/emotion-english-distilroberta-base`
* **Audio**: Fine-tuned `microsoft/wavlm-base-plus` for 5 emotions:

  * `anger`, `happy`, `neutral`, `sad`, `surprise`

---

## 🛠️ Setup Instructions

1. Clone the repository:

```bash
git clone <your-repo-url>
cd project_root
```

2. Create a virtual environment & install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

3. Set AWS credentials:

```bash
export AWS_ACCESS_KEY_ID=yourkey
export AWS_SECRET_ACCESS_KEY=yoursecret
```

4. Run the API:

```bash
uvicorn main:app --reload
```

---

## 🧪 Example Requests

**STT (Emotion from Audio):**

```bash
curl -X POST -F "file=@sample.wav" http://localhost:8000/stt/analyze
```

**TTS (Emotion-aware Speech Synthesis):**

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"text": "You are amazing! But I'm worried."}' \
     http://localhost:8000/generate-audio
```

---

## 👨‍💻 Author

Developed by İlhan Randa & Baturalp Kesici  for a thesis project exploring emotion-enhanced speech interfaces for accessibility.

---

## 📄 License

This project is part of academic research and intended for educational and experimental
![Image](https://github.com/user-attachments/assets/019ac77c-c0c8-4833-bc41-22e93754b694)

![Image](https://github.com/user-attachments/assets/4408e26b-c463-479b-922b-37f0fb2bd7a7)

![Image](https://github.com/user-attachments/assets/f458c60e-5b56-4a32-ac08-4e02547320be)
