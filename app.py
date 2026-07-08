from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input

app = Flask(__name__)

model = load_model("xception_scene.keras")

UPLOAD_FOLDER = "static/uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

class_names = [
    "Buildings",
    "Forest",
    "Glacier",
    "Mountain",
    "Sea",
    "Street"
]


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    image = cv2.imread(filepath)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (299,299)
    )

    image = image.astype(np.float32)

    image = preprocess_input(image)

    image = np.expand_dims(
        image,
        axis=0
    )

    prediction = model.predict(
        image,
        verbose=0
    )

    predicted_class = class_names[
        np.argmax(prediction)
    ]

    confidence = float(
        np.max(prediction)*100
    )

    return render_template(

        "result.html",

        prediction=predicted_class,

        confidence=f"{confidence:.2f}",

        image_path=filepath

    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=7860
    )