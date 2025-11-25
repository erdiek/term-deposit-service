import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Initialize the App
# This creates the web server. Title and version appear in the docs.
app = FastAPI(title="Term Deposit Prediction API", version="1.0")

# 2. Load the Model
# This file must exist in the same folder as this script.
try:
    model = joblib.load('model.joblib')
    print("Model loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model. {e}")
    model = None

# 3. Define the Data Contract
# Matches the columns in CSV exactly.
class CustomerData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str

# 4. Health Check Endpoint
# Check if the app is alive.
@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_loaded": True}

# 5. Prediction Endpoint
# Accept one customer's data and returns a prediction.
@app.post("/predict")
def predict(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Step A: Convert the input JSON to a DataFrame
        # We wrap data.dict() in a list [] to make a 1-row DataFrame
        input_df = pd.DataFrame([data.model_dump()])
        
        # Step B: Make the prediction
        # The pipeline handles scaling and one-hot encoding automatically!
        prediction = model.predict(input_df)[0]
        
        # Step C: Get probability (Confidence)
        # model.predict_proba returns [[prob_0, prob_1]]
        probability = model.predict_proba(input_df)[0][1]

        # Step D: Return JSON response
        return {
            "prediction": int(prediction),
            "label": "Subscribed" if prediction == 1 else "Not Subscribed",
            "probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))