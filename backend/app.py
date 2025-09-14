import base64
from typing import List
import os

os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from ultralytics import YOLO
from PIL import Image
import cv2

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


class PredictResult:
    clean: CarCleanliness
    damage: CarCondition
    dirtyImages: List[Image.Image]
    damagedImages: List[Image.Image]

    def __init__(self, clean, damage, dirtyImages=[], damagedImages=[]):
        self.damage = damage
        self.clean = clean
        self.dirtyImages = dirtyImages
        self.damagedImages = damagedImages


def format_data(sample, system_message):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample[0],
                },
                {
                    "type": "text",
                    "text": sample[1],
                },
            ],
        }
    ]


def analyze_detections(detections):
    # Initialize flags
    has_damage = False
    has_dirt = False

    # Check for damage-related classes
    damage_classes = ["dent", "scratch", "rust", "car-scratch"]
    dirt_classes = ["dirt"]

    for detection in detections:
        class_name = detection['class'].lower()
        confidence = detection['confidence']

        # Only consider detections with confidence >= 20%
        if confidence >= 0.2:
            if any(damage_class in class_name for damage_class in damage_classes):
                has_damage = True
            if any(dirt_class in class_name for dirt_class in dirt_classes):
                has_dirt = True

    # Determine verdicts
    damage_verdict = "damaged" if has_damage else "not damaged"
    dirt_verdict = "dirty" if has_dirt else "clean"

    return {
        "verdict": damage_verdict,
        "cleanliness": dirt_verdict
    }


def predict(dents_model, rust_dirt_model, scratches_model, images) -> PredictResult:
    # Load YOLO models
    model1 = YOLO(dents_model)
    model2 = YOLO(rust_dirt_model)
    model3 = YOLO(scratches_model)

    damaged = False
    dirty = False
    dirtyImages = []
    damagedImages = []

    def process_results(results, image, detections, dirtyImages, damagedImages, model_name):
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = result.names[class_id].lower()
                bbox = box.xyxy[0].tolist()
                confidence = box.conf.item()

                # Ignore low-confidence detections
                if confidence < 0.2:
                    continue

                detection = {
                    "model": model_name,
                    "class": class_name,
                    "bbox": bbox,
                    "confidence": confidence
                }
                detections.append(detection)

                crop = cropDetection(image.copy(), detection)
                if "dirt" in class_name:
                    dirtyImages.append(crop)
                else:
                    damagedImages.append(crop)

    for image in images:
        detections = []

        # Run predictions
        process_results(model1(image, verbose=False), image, detections, dirtyImages, damagedImages, "Model1 (Dents & Scratches)")
        process_results(model2(image, verbose=False), image, detections, dirtyImages, damagedImages, "Model2 (Rust, Dirt & Scratches)")
        process_results(model3(image, verbose=False), image, detections, dirtyImages, damagedImages, "Model3 (Scratches Only)")

        # Analyze detections
        analysis = analyze_detections(detections)

        print(f'"verdict": "{analysis["verdict"]}"')
        print(f'"cleanliness": "{analysis["cleanliness"]}"')

        # Build readable output
        text_output = "YOLO Results:\n"
        for detection in detections:
            text_output += (
                f"Model: {detection['model']}\n"
                f"Class: {detection['class']}\n"
                f"BBox: [{detection['bbox'][0]:.2f}, {detection['bbox'][1]:.2f}, "
                f"{detection['bbox'][2]:.2f}, {detection['bbox'][3]:.2f}]\n"
                f"Confidence: {detection['confidence']:.4f}\n"
                f"{'-' * 50}\n"
            )
        text_output += f"\nAnalysis Results:\n"
        text_output += f"Verdict: {analysis['verdict']}\n"
        text_output += f"Cleanliness: {analysis['cleanliness']}\n"
        print(text_output)

        if analysis["verdict"] == "damaged":
            damaged = True
        if analysis["cleanliness"] == "dirty":
            dirty = True

    return PredictResult(
        damage=CarCondition.Damaged if damaged else CarCondition.Good,
        clean=CarCleanliness.Dirty if dirty else CarCleanliness.Clean,
        dirtyImages=dirtyImages,
        damagedImages=damagedImages
    )



def toBase64(crop: Image.Image) -> str:
    buffer: io.BytesIO = io.BytesIO()
    crop.save(buffer, format="PNG")
    data: bytes = buffer.getvalue()
    return base64.b64encode(data).decode("utf-8")


def cropDetection(image: Image.Image, detection):
    x1, y1, x2, y2 = map(int, detection['bbox'])
    return image.crop((x1, y1, x2, y2))



@app.route('/api/evaluate', methods=["POST"])
def evaluate():
    if "carImages" not in request.files:
        return jsonify({"error": "No images"}), 400
    files = request.files.getlist("carImages")
    if len(files) == 0 or len(files) > 5:
        return jsonify({"error: too many / too little images"}), 400

    images = [Image.open(io.BytesIO(file.read())) for file in files]

    result = predict(dents_model="./models/dents.pt", rust_dirt_model="./models/rust_dirt.pt",
                     scratches_model="./models/scratches.pt", images=images)
    return jsonify(
        {
            "cleanliness": result.clean.name,
            "condition": result.damage.name,
            "dirtyCrops": [toBase64(image) for image in result.dirtyImages],
            "damagedCrops": [toBase64(image) for image in result.damagedImages]
        }
    ), 200


if __name__ == '__main__':
    print("Staring")
    app.run(host='0.0.0.0', port=8080)
