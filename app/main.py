from fastapi import FastAPI
from app.routes import tts_route

app = FastAPI()
app.include_router(tts_route.router)
