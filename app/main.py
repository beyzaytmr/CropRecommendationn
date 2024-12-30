from fastapi import FastAPI
from routers import crop

app = FastAPI()

# Ana rotayı tanımlayalım
@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Recommendation API"}

# Router'ı uygulamaya dahil edelim
app.include_router(crop.router)
