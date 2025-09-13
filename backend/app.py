import base64
from typing import List

from flask import Flask, request, jsonify
import io
from PIL import Image
from enum import Enum

from CarDamageDetector import CarDamageDetector


class CarCleanliness(Enum):
    Clean = 0
    Dirty = 1

class CarCondition(Enum):
    Good = 0,
    Damaged = 1


app = Flask(__name__)

def toBase64(crop: Image.Image) -> str:
    buffer: io.BytesIO = io.BytesIO()
    crop.save(buffer, format="PNG")
    data: bytes = buffer.getvalue()
    return base64.b64encode(data).decode("utf-8")

def cropDetections(image: Image.Image, detections):
    crops = []

    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        crop = image.crop((x1, y1, x2, y2))
        crops.append(crop)

    return crops


def evaluateCarCleanliness(images: List[Image.Image]) -> CarCleanliness:
    return CarCleanliness.Clean # TODO needs to be implemented by ML guys

def evaluateCarCondition(images: List[Image.Image]) -> list[Image.Image]:
    detector = CarDamageDetector("./models/best.pt")
    damagedPlaces = []
    for image in images:
        print("Seeking damage in image...")
        for detection in detector.detect_from_image(image):
            damagedPlaces.extend(cropDetections(image, detection))

    return damagedPlaces

def imageContainsPlateNumber(image: Image.Image) -> bool:
    return False # TODO needs to be implemented by ML guys


@app.route('/api/evaluate', methods=["POST"])
def evaluate():
    if "carImages" not in request.files:
        return jsonify({"error": "No images"}), 400
    files = request.files.getlist("carImages")
    if len(files) == 0 or len(files) > 5:
        return jsonify({"error: too many / too little images"}), 400

    images = [Image.open(io.BytesIO(file.read())) for file in files]

    for image in images:
        if imageContainsPlateNumber(image):
            return jsonify({"error: image contains confidential information"}), 400

    damagedPlaces = [toBase64(image) for image in evaluateCarCondition(images)]
    carCondition = CarCondition.Damaged if len(damagedPlaces) > 0 else CarCondition.Good

    return jsonify({"cleanliness": evaluateCarCleanliness(images).name, "condition": carCondition.name, "damagedPlaces": damagedPlaces}), 200



if __name__ == '__main__':
    print("Staring")
    app.run(host='0.0.0.0', port=8080)
