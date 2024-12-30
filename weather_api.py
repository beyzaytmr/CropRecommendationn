import requests

# OpenWeather API anahtarınızı buraya ekleyin
API_KEY = '17024ab839866a68487809463be3bbda'  # OpenWeather API Key

def get_weather(city):
    # OpenWeather API URL
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    # API'yi çağır
    response = requests.get(url)

    if response.status_code == 200:
        # API yanıtını JSON formatında alalım
        data = response.json()

        # Sıcaklık, nem ve yağış verilerini alalım
        temperature = data['main']['temp']  # Sıcaklık
        humidity = data['main']['humidity']  # Nem
        rainfall = data.get('rain', {}).get('1h', 0)  # Yağış (mm)

        # Sonuçları döndür
        return {
            "city": city,
            "temperature": f"{temperature}°C",
            "humidity": f"{humidity}%",
            "rainfall": f"{rainfall} mm"
        }
    else:
        # Hata durumunda mesaj döndür
        return {"error": f"Hava durumu bilgisi alınamadı. Durum kodu: {response.status_code}"}

# Şehir adı girerek fonksiyonu test edelim
if __name__ == "__main__":
    city_name = input("Hava durumu için bir şehir adı girin: ")
    result = get_weather(city_name)
    print(result)
