from fastapi import FastAPI

from api.translation.service import service as translation

app = FastAPI()


@app.get("/translate")
async def translate(message: str):
    return translation.translate(message)