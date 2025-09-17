from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# ✅ Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # or ["http://localhost:3000"] for stricter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load your trained model (you'll create fraud_model.joblib later)
model = joblib.load("fraud_model.joblib")

app = FastAPI()

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.post("/predict")
def predict(data: Transaction):
    features = np.array([[data.Time, data.V1, data.V2, data.V3, data.V4, data.V5,
                          data.V6, data.V7, data.V8, data.V9, data.V10, data.V11,
                          data.V12, data.V13, data.V14, data.V15, data.V16, data.V17,
                          data.V18, data.V19, data.V20, data.V21, data.V22, data.V23,
                          data.V24, data.V25, data.V26, data.V27, data.V28, data.Amount]])
    
    probability = model.predict_proba(features)[0][1]

    # ✅ Custom threshold (instead of 0.5)
    threshold = 0.3  
    prediction = 1 if probability >= threshold else 0

    return {"label": prediction, "probability": float(probability)}


    from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI()

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("fraud_model.joblib")

@app.get("/")
def home():
    return {"message": "✅ Fraud Detection Backend is running!"}

@app.post("/predict")
def predict(data: dict):
    features = np.array([list(data.values())]).astype(float)
    probability = model.predict_proba(features)[0][1]
    threshold = 0.3
    label = 1 if probability >= threshold else 0
    return {"label": label, "probability": float(probability)}


