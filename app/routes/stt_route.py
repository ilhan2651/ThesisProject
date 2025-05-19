# app/routes/stt_route.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from app.services.stt_service import process_audio_file

router = APIRouter(prefix="/stt", tags=["stt"])

@router.post("/analyze")
async def stt_analyze(file: UploadFile = File(...)):
    # 1) Klasörü hazırla
    out_dir = Path("ses_dosyalari")
    out_dir.mkdir(exist_ok=True)

    # 2) Dosyayı al ve kaydet
    target = out_dir / file.filename
    content = await file.read()
    target.write_bytes(content)

    # 3) İşle (transcribe + emotion)
    result = process_audio_file(str(target))
    if not result["text"]:
        raise HTTPException(status_code=400, detail="Transcription failed")

    return result
