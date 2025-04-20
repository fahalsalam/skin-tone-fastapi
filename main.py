from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
from typing import List
import joblib  # For model saving

app = FastAPI(title="Skin Tone ML API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "skin_tone_kmeans.joblib"

class SkinToneClassifier:
    def __init__(self):
        self.model = self.load_or_train_model()

    def load_or_train_model(self):
        if os.path.exists(MODEL_PATH):
            return joblib.load(MODEL_PATH)
        return KMeans(n_clusters=3, n_init=10)

    def save_model(self):
        joblib.dump(self.model, MODEL_PATH)

classifier = SkinToneClassifier()

def extract_skin(image: np.ndarray) -> np.ndarray:
    """Extract skin regions using HSV filtering"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(image, image, mask=mask), mask

@app.post("/analyze")
async def analyze_skin(file: UploadFile = File(..., description="Upload an image with face")):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are allowed")

    try:
        # Load image
        image = cv2.imdecode(
            np.frombuffer(await file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )
        if image is None:
            raise HTTPException(400, "Invalid image format")

        # Extract skin and mask
        skin, mask = extract_skin(image)
        skin_rgb = cv2.cvtColor(skin, cv2.COLOR_BGR2RGB)
        skin_pixels = skin_rgb[mask > 0]

        if len(skin_pixels) == 0:
            return {"skin_tone": "No skin detected"}

        # Reshape for KMeans
        skin_pixels = skin_pixels.reshape(-1, 3)

        # Train KMeans model
        classifier.model.fit(skin_pixels)
        classifier.save_model()

        # Get dominant color
        dominant_color = classifier.model.cluster_centers_[0]
        brightness = np.mean(dominant_color)

        # Tone classification
        tone = (
            "Very Light" if brightness > 200 else
            "Light" if brightness > 160 else
            "Medium" if brightness > 120 else
            "Dark"
        )

        # Confidence: largest cluster size ratio
        labels = classifier.model.labels_
        largest_cluster_count = np.bincount(labels).max()
        confidence = largest_cluster_count / len(labels)

        return {
            "skin_tone": tone,
            "dominant_color": dominant_color.tolist(),
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")

@app.api_route("/", methods=["GET", "HEAD"])
async def health_check(request: Request):
    return {"status": "Healthy", "model": "KMeans"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
