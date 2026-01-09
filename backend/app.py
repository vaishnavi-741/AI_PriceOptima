from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("priceoptima_lgb_model.joblib")

# Create a request model
class PredictRequest(BaseModel):
    cost: float
    stock: int
    day: int
    week: int
    month: int
    is_weekend: int
    competitor: float
    discount: float
    holiday: int
    last_week_sales: int
    category: int

@app.post("/predict")
def predict(request: PredictRequest):
    data = request.dict()

    df = pd.DataFrame([data])

    demand = model.predict(df)[0]
    price = data["cost"] * (1 + demand / 1000)

    return {
        "predicted_demand": round(demand, 2),
        "recommended_price": round(price, 2)
    }
