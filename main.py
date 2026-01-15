from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from model import predict

app = FastAPI()

class Image(BaseModel):
    pixels: list  # length 784

@app.post("/predict")
def predict_digit(img: Image):
    x = np.array(img.pixels)
    digit, confidence = predict(x)
    return {
        "digit": digit,
        "confidence": confidence
    }