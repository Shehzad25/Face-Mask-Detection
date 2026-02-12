# inference/model.py

import cv2
import os
from tensorflow.keras.models import load_model

# Absolute path to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correct model path inside Docker and locally
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "mask_detector_model.h5")

def load_mask_model():
    return load_model(MODEL_PATH, compile=False)

def load_face_detector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

