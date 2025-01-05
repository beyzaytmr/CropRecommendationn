import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Modeli dışa aktarmak için

# Veriyi yükleme
df = pd.read_csv('data/Crop_Recommendation.csv')

# Özellikleri ve hedefi ayırma
X = df[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity',
        'pH_Value', 'Rainfall']]

encoder = LabelEncoder()
y = encoder.fit_transform(df['Crop'])

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=28, test_size=0.2)

# Modeli eğitme
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Modeli değerlendirme
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion Matrix görselleştirme
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
#plt.show()

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Modeli ve encoder'ı dışa aktarma
joblib.dump(model, 'crop_recommendation_model.pkl')
joblib.dump(encoder, 'label_encoder.pkl')
print("Model ve encoder başarıyla dışa aktarıldı.")


# Yeni verilerle tahmin için fonksiyon
def predict_crop(input_features):
    # Dışa aktarılan modeli ve encoder'ı yükleme
    loaded_model = joblib.load('crop_recommendation_model.pkl')
    loaded_encoder = joblib.load('label_encoder.pkl')

    # Özellikleri DataFrame formatına çevirme
    input_df = pd.DataFrame([input_features])

    # Tahmin yapma
    prediction_encoded = loaded_model.predict(input_df)
    prediction_decoded = loaded_encoder.inverse_transform(prediction_encoded)
    return prediction_decoded[0]


# Örnek kullanım
example_features = {
    'Nitrogen': 150,
    'Phosphorus': 55,
    'Potassium': 12,
    'Temperature': 19.45591848,
    'Humidity': 18.02235902,
    'pH_Value': 8.423873703,
    'Rainfall': 78.44910564
}
predicted_crop = predict_crop(example_features)
print(f"Burada yetiştirilebilecek en uygun ürün: {predicted_crop}")
