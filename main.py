from fastapi import FastAPI, Body
from model import SentimentInference


app = FastAPI()
inf = SentimentInference.load('f')


@app.post("/sentiment")
async def infer_sentiment(body:str =  Body(...)):
    res = inf(body)
    response = "negative" if res == 0 else "positive"
    return {"sentiment": response}