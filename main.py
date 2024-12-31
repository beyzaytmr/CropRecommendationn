from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import pickle
import numpy as np

# FastAPI Uygulaması
app = FastAPI()

# OpenWeather API Anahtarı
API_KEY = "17024ab839866a68487809463be3bbda"

# Model Yükleme
try:
    with open("app/models/crop_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("Model dosyası bulunamadı. Lütfen doğru yolu kontrol edin.")

# Veri Modeli (Tahmin için)
class CropPredictionRequest(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


@app.get("/weather")
def get_weather(city: str):
    """
    Şehir adına göre hava durumu bilgilerini döndüren API
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    # OpenWeather API Çağrısı
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        temperature = data['main']['temp']  # Sıcaklık
        humidity = data['main']['humidity']  # Nem
        rainfall = data.get('rain', {}).get('1h', 0)  # Yağış miktarı

        return {
            "city": city,
            "temperature": f"{temperature}°C",
            "humidity": f"{humidity}%",
            "rainfall": f"{rainfall} mm"
        }
    else:
        raise HTTPException(status_code=response.status_code, detail="Hava durumu bilgisi alınamadı.")


@app.post("/predict")
def predict_crop(data: CropPredictionRequest):
    """
    Tarım ürünü tahmini yapan API
    """
    # Girdi verilerini hazırlama
    input_data = np.array([
        data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall
    ]).reshape(1, -1)

    # Tahmin yap
    try:
        prediction = model.predict(input_data)[0]
        return {
            "prediction": prediction,
            "input": {
                "N": data.N,
                "P": data.P,
                "K": data.K,
                "temperature": data.temperature,
                "humidity": data.humidity,
                "ph": data.ph,
                "rainfall": data.rainfall
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin yapılırken bir hata oluştu: {str(e)}")

# Ana Sayfa
@app.get("/")
def root():
    return {"message": "Crop Prediction and Weather API is running. Access /docs for API documentation."}
