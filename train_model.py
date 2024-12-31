from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

# Veri setini yükleyelim
df = pd.read_csv("data/Crop_recommendation.csv")

# Veri setindeki ilk birkaç satırı kontrol edelim (opsiyonel)
print("Veri setindeki ilk 5 satır:")
print(df.head())

# Özellikler ve hedef sütun
X = df.drop("label", axis=1)  # Özellikler (N, P, K, sıcaklık, nem, pH, yağış)
y = df["label"]  # Tahmin edilecek hedef (ürün türü)

# Modeli eğitme
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Modeli kaydetme
with open("app/models/crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model başarıyla eğitildi ve 'crop_model.pkl' dosyasına kaydedildi!")
