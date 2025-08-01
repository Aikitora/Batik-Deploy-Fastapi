from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import logging

from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1️⃣ Inisialisasi FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # atau ganti dengan ["http://localhost:5173"] untuk lebih aman
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2️⃣ Load model Keras with error handling
model = None
try:
    # Try loading the genetic algorithm model first
    model = tf.keras.models.load_model("final_tuned_genetic_algorithm_model.keras")
    logger.info("Successfully loaded final_tuned_genetic_algorithm_model.keras")
except Exception as e:
    logger.warning(f"Failed to load genetic algorithm model: {e}")
    try:
        # Fallback to the regular tuned model
        model = tf.keras.models.load_model("final_tuned_model.keras")
        logger.info("Successfully loaded final_tuned_model.keras")
    except Exception as e2:
        logger.error(f"Failed to load both models: {e2}")
        raise Exception("Unable to load any model files. Please check if the model files exist and are compatible.")

if model is None:
    raise Exception("No model could be loaded. Please ensure model files are present.")

# 3️⃣ Load labels dari file
try:
    with open("labels.txt") as f:
        labels = [line.strip() for line in f]
    logger.info(f"Loaded {len(labels)} labels successfully")
except Exception as e:
    logger.error(f"Failed to load labels: {e}")
    raise Exception("Unable to load labels.txt file")

# 4️⃣ Ukuran input gambar (samakan dengan training!)
IMAGE_SIZE = (160, 160)  # contoh, sesuaikan dengan model Anda

# 5️⃣ Endpoint root
@app.get("/")
def read_root():
    return {"message": "FastAPI Batik Classifier is running!"}

# 6️⃣ Endpoint prediksi (upload image)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "File must be an image"}
            )
        
        image = Image.open(file.file).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)[0]
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
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Prediction failed: {str(e)}"}
        )
