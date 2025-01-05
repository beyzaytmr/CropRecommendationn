import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Veriyi yükleme
data_path = os.path.join(os.path.dirname(__file__), 'data', 'Crop_Recommendation.csv')
df = pd.read_csv(data_path)

# Özellikleri ve hedefi ayırma
X = df[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']]
encoder = LabelEncoder()
y = encoder.fit_transform(df['Crop'])

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

# Modeli eğitme
model = RandomForestClassifier(n_estimators=100, random_state=28)
model.fit(X_train, y_train)

# Modeli değerlendirme
y_pred = model.predict(X_test)
print("\nModel Değerlendirme Sonuçları:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Confusion Matrix görselleştirme
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Confusion Matrix görselini kaydedin
plt.close()

# Modeli ve encoder'ı kaydetme
model_path = os.path.join(os.path.dirname(__file__), 'crop_recommendation_model.pkl')
encoder_path = os.path.join(os.path.dirname(__file__), 'label_encoder.pkl')

joblib.dump(model, model_path)
joblib.dump(encoder, encoder_path)
print(f"Model ve encoder başarıyla kaydedildi: {model_path}, {encoder_path}")

# Yeni verilerle tahmin için fonksiyon
def predict_crop(input_features):
    # Dışa aktarılan modeli ve encoder'ı yükleme
    loaded_model = joblib.load(model_path)
    loaded_encoder = joblib.load(encoder_path)

    # Özellikleri DataFrame formatına çevirme
    input_df = pd.DataFrame([input_features])

    # Tahmin yapma
    prediction_encoded = loaded_model.predict(input_df)
    prediction_decoded = loaded_encoder.inverse_transform(prediction_encoded)
    return prediction_decoded[0]

# Örnek kullanım
example_features = {
    'Nitrogen': 100,
    'Phosphorus': 40,
    'Potassium': 50,
    'Temperature': 22.0,
    'Humidity': 80.0,
    'pH_Value': 6.5,
    'Rainfall': 200.0
}
predicted_crop = predict_crop(example_features)
print(f"\nÖnerilen ürün: {predicted_crop}")
