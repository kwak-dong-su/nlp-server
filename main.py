from fastapi import FastAPI

from api.translation.request.request import Message
from api.translation.service import service as translation

app = FastAPI()


@app.get("/translate")
async def translate(message: Message):
    return translation.translate(message.content)