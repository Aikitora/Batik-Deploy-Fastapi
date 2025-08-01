from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image

# 1️⃣ Inisialisasi FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # atau ganti dengan ["http://localhost:5173"] untuk lebih aman
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2️⃣ Load model Keras
model = tf.keras.models.load_model("final_tuned_model.keras")

# 3️⃣ Load labels dari file
with open("labels.txt") as f:
    labels = [line.strip() for line in f]

# 4️⃣ Ukuran input gambar (samakan dengan training!)
IMAGE_SIZE = (224, 224)  # contoh, sesuaikan dengan model Anda

# 5️⃣ Endpoint root
@app.get("/")
def read_root():
    return {"message": "FastAPI Batik Classifier is running!"}

# 6️⃣ Endpoint prediksi (upload image)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)[0]
    prediction_list = [
        {"label": labels[i], "confidence": float(pred)}
        for i, pred in enumerate(predictions)
    ]
    prediction_list.sort(key=lambda x: x["confidence"], reverse=True)
    top_predictions = prediction_list[:5]
    top_prediction = top_predictions[0]

    return {
        "success": True,
        "data": {
            "class_name": top_prediction["label"],
            "confidence": top_prediction["confidence"],
            "probabilities": {
                p["label"]: p["confidence"] for p in top_predictions
            }
        }
    }
