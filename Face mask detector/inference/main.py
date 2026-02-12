import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from inference.model import load_mask_model, load_face_detector
from inference.utils import predict_mask

app = FastAPI(title="Face Mask Detection API")

model = load_mask_model()
face_cascade = load_face_detector()


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    results = []

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        label, confidence = predict_mask(model, face)

        results.append({
            "label": label,
            "confidence": round(confidence, 3)
        })

    return {
        "faces_detected": len(results),
        "predictions": results
    }
