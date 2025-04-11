from fastapi import APIRouter
from models.request_models import TTSRequest
from app.services.tts_helper import text_to_speech_and_emotion
from fastapi.responses import FileResponse
import uuid
import os

router = APIRouter()

@router.post("/generate-audio")
def generate_audio(request: TTSRequest):
    filename = f"speech_{uuid.uuid4().hex}.mp3"
    output_path = os.path.join("ses_dosyalari", filename)

    result = text_to_speech_and_emotion(request.text, output_file=output_path)

    if result:
        return FileResponse(output_path, media_type="audio/mpeg", filename=filename)
    return {"error": "Ses üretilemedi."}
