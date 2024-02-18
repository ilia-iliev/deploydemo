from fastapi import FastAPI
from model import SentimentInference
from pydantic import BaseModel


class Review(BaseModel):
    text: str
    username: str | None = None

app = FastAPI()
inf = SentimentInference.load()


@app.post("/sentiment")
async def infer_sentiment(req: Review):
    res = inf(req.text)
    response = "negative" if res == 0 else "positive"
    return {"sentiment": response}
