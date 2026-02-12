import cv2
import numpy as np


IMG_SIZE = 128
THRESHOLD = 0.8


def preprocess_face(face):
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face / 255.0
    face = np.reshape(face, (1, IMG_SIZE, IMG_SIZE, 3))
    return face


def predict_mask(model, face):
    face = preprocess_face(face)
    pred = model.predict(face, verbose=0)[0][0]

    if pred > THRESHOLD:
        return "Mask", float(pred)
    else:
        return "No Mask", float(1 - pred)
