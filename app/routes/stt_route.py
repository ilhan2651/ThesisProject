# app/routes/stt_route.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from app.others.Stt_Emo import transcribe_audio

router = APIRouter(
    prefix="/stt",
    tags=["stt"]
)

@router.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    # 1) Kayıt klasörünü hazırla
    out_dir = Path("ses_dosyalari")
    out_dir.mkdir(exist_ok=True)

    # 2) Gelen dosyayı kaydet
    file_path = out_dir / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    # 3) Stt işlemini yap
    text = transcribe_audio(str(file_path))
    if text is None:
        raise HTTPException(status_code=400, detail="Transcription failed")

    # 4) JSON olarak metni dön
    return {"text": text}
