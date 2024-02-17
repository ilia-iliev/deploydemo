from fastapi import FastAPI
from model import SentimentInference


app = FastAPI()
inf = SentimentInference.load('f')


@app.get("/{model}")
async def infer_sentiment(film_review: str):
    res = inf(film_review)
    response = "negative" if res == 0 else "positive"
    return {"sentiment": response}