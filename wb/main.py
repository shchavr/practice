from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler  

app = FastAPI()

try:
    model = joblib.load("models/best.pkl")
except Exception as e:
    raise RuntimeError(f"Ошибка загрузки модели или скейлера: {e}")

class InputData(BaseModel):
    danceability: float  
    energy: float
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    duration_min: float  
    dance_energy_rat: float 
    mode_1: float  
    time_signature_1: float 
    time_signature_3: float
    time_signature_4: float
    time_signature_5: float  
    explicit_1: float 
    category_track_genre: float  

@app.post("/predict")
def predict(data: InputData):
    try:
        
        input_df = pd.DataFrame([data.dict()])

        prediction = model.predict(input_df) #
        return {"predicted_value": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
