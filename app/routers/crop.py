from fastapi import APIRouter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

router = APIRouter()

# Veri seti ve model dosyalarını yükleyelim
df = pd.read_csv("data/Crop_recommendation.csv")
model = pickle.load(open("models/crop_model.pkl", "rb"))

@router.post("/recommend")
def recommend_crop(N: int, P: int, K: int, temperature: float, humidity: float, pH: float, rainfall: float):
    # Girdileri modele uygun hale getirelim
    input_data = [[N, P, K, temperature, humidity, pH, rainfall]]
    prediction = model.predict(input_data)
    return {"recommended_crop": prediction[0]}
