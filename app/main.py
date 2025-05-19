from fastapi import FastAPI
from app.routes.tts_route import router as tts_router
from app.routes.stt_route import router as stt_router

app = FastAPI()

# app/main.py




app = FastAPI()

# Mevcut TTS route
app.include_router(tts_router)

# Yeni STT route
app.include_router(stt_router)
