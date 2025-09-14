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

    def __init__(self, clean, damage):
        self.damage = damage
        self.clean = clean



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
    # Set local paths
    LOCAL_MODEL_PATH = "qwen2_vl_7b_instruct"
    LOCAL_PROCESSOR_PATH = "qwen2_vl_7b_instruct_processor"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    EPOCHS = 1
    BATCH_SIZE = 1
    GRADIENT_CHECKPOINTING = True
    USE_REENTRANT = False
    OPTIM = "paged_adamw_32bit"
    LEARNING_RATE = 2e-5
    LOGGING_STEPS = 50
    EVAL_STEPS = 50
    SAVE_STEPS = 50
    EVAL_STRATEGY = "steps"
    SAVE_STRATEGY = "steps"
    METRIC_FOR_BEST_MODEL = "eval_loss"
    LOAD_BEST_MODEL_AT_END = True
    MAX_GRAD_NORM = 1
    WARMUP_STEPS = 0
    DATASET_KWARGS = {"skip_prepare_dataset": True}
    REMOVE_UNUSED_COLUMNS = False
    MAX_SEQ_LEN = 128
    NUM_STEPS = (283 // BATCH_SIZE) * EPOCHS

    system_message = """You are an assistant that helps detecting dents, scratches, dirt, rust and other damages of a car via photo, you will be provided results of three YOLO model as important hints to make a comment more precise. YOLO are trained to detect dents, scratches, rust and dirt, so if you spot something different than that please add it as acomment but focus on describing YOLO results. The final format of output should be: comment: location of damage/dirt, degree of damage or dirt or other defect. The target of your comment is a user, so try to be precise, short and exclude all sophisticated terms."""

    # Load your models
    model1 = YOLO(dents_model)
    model2 = YOLO(rust_dirt_model)
    model3 = YOLO(scratches_model)

    # Process each image
    for image in images:

        # Run predictions with all three models
        results1 = model1(image, verbose=False)
        results2 = model2(image, verbose=False)
        results3 = model3(image, verbose=False)

        # Initialize a list to store detection information
        detections = []

        # Process results from model1
        for result in results1:
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = result.names[class_id]
                bbox = box.xyxy[0].tolist()
                confidence = box.conf.item()
                detections.append({
                    "model": "Model1 (Dents & Scratches)",
                    "class": class_name,
                    "bbox": bbox,
                    "confidence": confidence
                })

        # Process results from model2
        for result in results2:
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = result.names[class_id]
                bbox = box.xyxy[0].tolist()
                confidence = box.conf.item()
                detections.append({
                    "model": "Model2 (Rust, Dirt & Scratches)",
                    "class": class_name,
                    "bbox": bbox,
                    "confidence": confidence
                })

        # Process results from model3
        for result in results3:
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = result.names[class_id]
                bbox = box.xyxy[0].tolist()
                confidence = box.conf.item()
                detections.append({
                    "model": "Model3 (Scratches Only)",
                    "class": class_name,
                    "bbox": bbox,
                    "confidence": confidence
                })

        # Analyze the detections
        analysis = analyze_detections(detections)

        # Print the results
        print(f'"verdict": "{analysis["verdict"]}"')
        print(f'"cleanliness": "{analysis["cleanliness"]}"')

        # Continue with your existing text_output generation
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

        # You can also add the verdict to your text output if desired
        text_output += f"\nAnalysis Results:\n"
        text_output += f"Verdict: {analysis['verdict']}\n"
        text_output += f"Cleanliness: {analysis['cleanliness']}\n"

        print(text_output)
        res = PredictResult(damage=CarCondition.Damaged if analysis["verdict"] == "damaged" else CarCondition.Good, clean=CarCleanliness.Clean if analysis["cleanliness"] == "clean" else CarCleanliness.Dirty)
        return res


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


def evaluateCarCleanliness(images: List[Image.Image]) -> list[Image.Image]:
    detector = CarDamageDetector("./models/best.pt")
    dirtyPlaces = []
    for image in images:
        print("Seeking damage in image...")
        for detection in detector.detect_from_image(image):
            dirtyPlaces.extend(cropDetections(image, detection))

    return dirtyPlaces


def evaluateCarCondition(images: List[Image.Image]) -> list[Image.Image]:
    detector = CarDamageDetector("./models/best.pt")
    damagedPlaces = []
    for image in images:
        print("Seeking damage in image...")
        for detection in detector.detect_from_image(image):
            damagedPlaces.extend(cropDetections(image, detection))

    return damagedPlaces


@app.route('/api/evaluate', methods=["POST"])
def evaluate():
    if "carImages" not in request.files:
        return jsonify({"error": "No images"}), 400
    files = request.files.getlist("carImages")
    if len(files) == 0 or len(files) > 5:
        return jsonify({"error: too many / too little images"}), 400

    images = [Image.open(io.BytesIO(file.read())) for file in files]

    result = predict(dents_model="./models/dents.pt", rust_dirt_model="./models/rust_dirt.pt", scratches_model="./models/scratches.pt", images = images)
    return jsonify({"cleanliness": result.clean.name, "condition": result.damage.name}), 200


if __name__ == '__main__':
    print("Staring")
    app.run(host='0.0.0.0', port=8080)
