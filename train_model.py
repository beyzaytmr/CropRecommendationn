from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

# Veri setini yükleyelim
df = pd.read_csv("data/Crop_recommendation.csv")

# İlk 10 satırı görüntüleyelim
print("Veri setindeki ilk 10 satır:")
print(df.head(10))

# Sütun isimlerini görüntüleyelim
print("\nSütunlar:")
print(df.columns)

# Veri setinin genel özet bilgisi
print("\nVeri setinin özet bilgisi:")
print(df.describe())

# Özellikler ve hedef sütun
X = df.drop("label", axis=1)
y = df["label"]

# Modeli eğitme
model = RandomForestClassifier()
model.fit(X, y)

# Modeli kaydetme
with open("app/models/crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model başarıyla eğitildi ve kaydedildi!")
