from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ml_model import analyze_trends, predict_total

app = FastAPI()

class TransactionData(BaseModel):
    user_id: int
    transactions: list[dict]

class TotalPredictionRequest(BaseModel):
    user_id: int
    accounts: list[dict]
    months: int

@app.post("/analyze_trends")
async def analyze_trends_endpoint(data: TransactionData):
    try:
        trends = analyze_trends(data.transactions)
        return {"user_id": data.user_id, "trends": trends}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_total")
async def predict_total_endpoint(data: TotalPredictionRequest):
    try:
        prediction = predict_total(data.user_id, data.accounts, data.months)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))