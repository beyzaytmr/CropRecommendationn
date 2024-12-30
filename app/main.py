from fastapi import FastAPI
import requests

app = FastAPI()

# OpenWeather API anahtarınızı buraya ekleyin
API_KEY = '17024ab839866a68487809463be3bbda'


@app.get("/weather")
def get_weather(city: str):
    """
    Şehir adına göre hava durumu bilgilerini döndüren API
    """
    # OpenWeather API URL
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    # API'yi çağır
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        # Hava durumu bilgilerini alalım
        temperature = data['main']['temp']  # Sıcaklık
        humidity = data['main']['humidity']  # Nem
        rainfall = data.get('rain', {}).get('1h', 0)  # Yağış

        return {
            "city": city,
            "temperature": f"{temperature}°C",
            "humidity": f"{humidity}%",
            "rainfall": f"{rainfall} mm"
        }
    else:
        # Hata durumunda mesaj döndür
        return {"error": f"Hava durumu bilgisi alınamadı. HTTP Durum Kodu: {response.status_code}"}
