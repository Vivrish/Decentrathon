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


def evaluateCarCleanliness(images: List[Image.Image]) -> CarCleanliness:
    return CarCleanliness.Clean # TODO needs to be implemented by ML guys

def evaluateCarCondition(images: List[Image.Image]) -> CarCondition:
    detector = CarDamageDetector("./models/best.pt")
    for image in images:
        if detector.detect_from_image(image):
            return CarCondition.Damaged
    return CarCondition.Good

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

    return jsonify({"cleanliness": evaluateCarCleanliness(images).name, "condition": evaluateCarCondition(images).name}), 200



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
