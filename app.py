from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from flask_cors import CORS

# Flask uygulamasını başlat
app = Flask(__name__)

# CORS'u etkinleştir
CORS(app)

# Model ve encoder'ı yükleme
model = joblib.load('crop_recommendation_model.pkl')
encoder = joblib.load('label_encoder.pkl')

# Ana sayfayı göstermek için route
@app.route('/')
def index():
    return render_template('index.html')  # templates/index.html dosyasını render eder

# Tahmin API'si
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Kullanıcıdan gelen veriyi al
        data = request.get_json()

        # Beklenen veri formatını kontrol et
        required_features = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
        for feature in required_features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400

        # Veriyi DataFrame formatına çevir
        input_df = pd.DataFrame([data])

        # Tahmin yap
        prediction_encoded = model.predict(input_df)
        prediction_decoded = encoder.inverse_transform(prediction_encoded)

        # Tahmin sonucunu döndür
        return jsonify({'predicted_crop': prediction_decoded[0]}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
