from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ml_model import analyze_trends

app = FastAPI()

class TransactionData(BaseModel):
    user_id: int
    transactions: list[dict]

@app.post("/analyze_trends")
async def analyze_trends_endpoint(data: TransactionData):
    try:
        trends = analyze_trends(data.transactions)
        return {"user_id": data.user_id, "trends": trends}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))