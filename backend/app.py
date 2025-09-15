import base64
import sys
from typing import List
import os

import numpy as np

os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from ultralytics import YOLO
import cv2

from flask import Flask, request, jsonify
import io
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
    dirtyImages: List[np.ndarray]
    damagedImages: List[np.ndarray]

    def __init__(self, clean, damage, dirtyImages=[], damagedImages=[]):
        self.damage = damage
        self.clean = clean
        self.dirtyImages = dirtyImages
        self.damagedImages = damagedImages


def preprocess_car_image(image, model_path='yolov8m-seg.pt', conf_threshold=0.25,
                         target_size=(640, 640), min_coverage=30):
    """
    Preprocess an image to extract and resize the largest car with mask coordinates

    Args:
        image: Input image
        model_path: Path to YOLO segmentation model
        conf_threshold: Confidence threshold for detection
        target_size: Target size for the resized image (width, height)
        min_coverage: Minimum percentage of image area the car should cover

    Returns:
        dict: Contains processed image and mask information
            - 'cropped_resized': Cropped and resized image of the largest car
            - 'mask_coordinates': Coordinates of the mask relative to the original image
            - 'bounding_box': Bounding box coordinates in the original image (x1, y1, x2, y2)
            - 'coverage_percentage': Percentage of image covered by the car
            - 'success': Boolean indicating if preprocessing was successful
            - 'message': Status message
    """

    # Initialize result dictionary
    result = {
        'cropped_resized': None,
        'mask_coordinates': None,
        'bounding_box': None,
        'coverage_percentage': 0,
        'success': False,
        'message': ''
    }

    try:
        # Load YOLOv8 segmentation model
        model = YOLO(model_path)

        # Perform inference
        results = model(image, conf=conf_threshold, verbose=False)

        if image is None:
            result['message'] = f"Error: Could not read image from {image}"
            return result, True

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()

        # Extract masks and process results
        car_info = []  # Store info about each detected car

        if results[0].masks is not None:
            for i, (mask, box) in enumerate(zip(results[0].masks, results[0].boxes)):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])

                # Filter for cars only (class_id 2 in COCO dataset)
                if class_name == 'car' and confidence > conf_threshold:
                    # Convert mask to numpy array
                    mask_data = mask.data[0].cpu().numpy()
                    mask_resized = cv2.resize(mask_data, (image.shape[1], image.shape[0]))

                    # Create binary mask
                    binary_mask = (mask_resized > 0.5).astype(np.uint8)

                    # Calculate mask area
                    mask_area = np.sum(binary_mask)

                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                    car_info.append({
                        'mask': binary_mask,
                        'area': mask_area,
                        'box': (x1, y1, x2, y2)
                    })

        # Check if any cars were detected
        if not car_info:
            result['message'] = "Error: No cars detected"
            return result, True

        # Find the largest car
        car_info.sort(key=lambda x: x['area'], reverse=True)
        largest_car = car_info[0]

        # Calculate percentage of image covered by the largest car
        image_area = image.shape[0] * image.shape[1]
        coverage_percentage = (largest_car['area'] / image_area) * 100
        result['coverage_percentage'] = coverage_percentage

        # Check if the car meets the minimum coverage requirement
        if coverage_percentage < min_coverage:
            result[
                'message'] = f"Error: Largest car covers only {coverage_percentage:.2f}% of the image (min {min_coverage}% required)"
            return result, False

        # Get bounding box coordinates
        x1, y1, x2, y2 = largest_car['box']
        result['bounding_box'] = (x1, y1, x2, y2)

        # Crop the image to the bounding box
        cropped = image[y1:y2, x1:x2]

        # Resize to target size
        cropped_resized = cv2.resize(cropped, target_size)
        result['cropped_resized'] = cropped_resized

        # Get mask coordinates (relative to original image)
        mask = largest_car['mask']
        y_indices, x_indices = np.where(mask == 1)
        result['mask_coordinates'] = list(zip(x_indices, y_indices))

        result['success'] = True
        result['message'] = f"Success: Preprocessed car covering {coverage_percentage:.2f}% of the image"

    except Exception as e:
        result['message'] = f"Error during processing: {str(e)}"

    return result, False


def load_models_from_file(models_file_path):
    """
    Load model paths from a text file
    Expected format: one model path per line, optionally with name:path format
    """
    models = {}
    try:
        with open(models_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                if ':' in line:
                    # Format: model_name:path/to/model.pt
                    parts = line.split(':', 1)
                    model_name = parts[0].strip()
                    model_path = parts[1].strip()
                else:
                    # Format: path/to/model.pt (use filename as name)
                    model_path = line
                    model_name = os.path.splitext(os.path.basename(model_path))[0]

                models[model_name] = model_path

        return models

    except FileNotFoundError:
        print(f"Error: Models file '{models_file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading models file: {e}")
        sys.exit(1)


def preprocess_image(image_path, output_size=(640, 640)):
    """
    Preprocess image according to the specified pipeline:
    1. Auto-orient (if needed)
    2. Resize: Stretch to 640x640
    3. Grayscale
    4. Auto-adjust contrast: Adaptive Equalization
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    # 1. Auto-orient (OpenCV loads images in BGR format, no orientation issues typically)
    # For orientation issues, you might need additional logic if images come from mobile devices

    # 2. Resize: Stretch to specified size
    img_resized = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)

    # 3. Convert to grayscale
    # img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 4. Auto-adjust contrast using Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # img_contrast = clahe.apply(img_gray)
    # img_contrast = clahe.apply(img_resized)

    return img_resized


def load_class_names_from_file(classes_file_path):
    """
    Load class names from a text file
    Expected format: one class name per line, optionally with confidence threshold
    """
    class_names = []
    class_thresholds = {}
    try:
        with open(classes_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Check if line contains threshold: class_name:threshold
                if ':' in line and not line.startswith(('http://', 'https://', '/', './', '../')):
                    parts = line.split(':', 1)
                    class_name = parts[0].strip()
                    threshold_part = parts[1].strip()

                    try:
                        threshold = float(threshold_part)
                        class_thresholds[class_name.lower()] = threshold
                        class_names.append(class_name)
                    except ValueError:
                        class_names.append(line)
                else:
                    class_names.append(line)

        return class_names, class_thresholds

    except FileNotFoundError:
        print(f"Error: Classes file '{classes_file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading classes file: {e}")
        sys.exit(1)


def parse_class_thresholds(threshold_string):
    """
    Parse class-specific thresholds from string format: class1:0.3,class2:0.5,class3:0.2
    """
    thresholds = {}
    if threshold_string:
        for item in threshold_string.split(','):
            item = item.strip()
            if ':' in item:
                class_name, threshold = item.split(':', 1)
                class_name = class_name.strip().lower()
                try:
                    thresholds[class_name] = float(threshold.strip())
                except ValueError:
                    print(f"Warning: Invalid threshold value for class '{class_name}': {threshold}")
    return thresholds


def is_box_center_in_mask(bbox, mask_coordinates):
    """
    Check if the center of a bounding box is inside mask coordinates

    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        mask_coordinates: List of (x, y) coordinates that form the mask

    Returns:
        bool: True if the center point is inside the mask, False otherwise
    """
    # Calculate the center point of the bounding box
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Convert mask coordinates to a contour for point-in-polygon test
    # Note: This creates a convex hull from the mask points
    if len(mask_coordinates) < 3:
        return False  # Need at least 3 points to form a polygon

    # Create a contour from mask coordinates
    contour = np.array(mask_coordinates, dtype=np.int32)

    # Use OpenCV's pointPolygonTest to check if center is inside the contour
    result = cv2.pointPolygonTest(contour, (center_x, center_y), False)

    # Return True if the point is inside or on the edge of the contour
    return result >= 0


def predict(images) -> PredictResult:
    models_dict = load_models_from_file("models.txt")
    all_class_names, file_class_thresholds = load_class_names_from_file("classes.txt")
    print("DEBUG all_class_names:", all_class_names, type(all_class_names))

    # Parse dirt classes - all other classes will be considered damage classes
    dirt_classes = ["dirt"]

    # Damage classes are all classes that are NOT dirt classes
    damage_classes = [cls for cls in all_class_names if cls.lower() not in dirt_classes]

    # Parse class-specific thresholds from command line
    cmd_class_thresholds = parse_class_thresholds('')

    # Combine thresholds from file and command line (command line takes precedence)
    class_thresholds = {**file_class_thresholds, **cmd_class_thresholds}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load all YOLO models
    models = {}
    for model_name, model_path in models_dict.items():
        try:
            models[model_name] = YOLO(model_path)
        except Exception as e:
            print(f"Error loading model {model_name} from {model_path}: {e}")
            continue

    if not models:
        print("No models loaded successfully. Exiting.")
        sys.exit(1)

    # Process each image
    for image in images:

        # Load the image
        mask_coords = []
        try:
            result, sk = preprocess_car_image(image)
            if sk:
                input_image = preprocess_image(image)
            else:
                input_image = result['cropped_resized']
                mask_coords = result['mask_coordinates'] or []
        except Exception as e:
            print(f"Error loading image")
            continue

        # Initialize a list to store detection information
        detections = []
        confident_detections = []

        damagedCrops = []
        dirtyCrops = []
        # Run predictions with all models
        for model_name, model in models.items():
            try:
                results = model(input_image, verbose=False)

                # Process results from each model
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        # Use provided class names if available, otherwise fall back to model's names
                        if hasattr(result, 'names') and result.names:
                            class_name = result.names[class_id]
                        elif class_id < len(all_class_names):
                            class_name = all_class_names[class_id]
                        else:
                            class_name = f"class_{class_id}"

                        bbox = box.xyxy[0].tolist()
                        confidence = box.conf.item()
                        if is_box_center_in_mask(bbox, mask_coords) or sk:

                            # Get class-specific threshold or use default
                            class_threshold = class_thresholds.get(class_name.lower(), 0.2)

                            detection_info = {
                                "model": model_name,
                                "class": class_name,
                                "bbox": bbox,
                                "confidence": confidence,
                                "threshold": class_threshold,
                                "above_threshold": confidence >= class_threshold
                            }

                            detections.append(detection_info)

                            if class_name.lower() == "dirt":
                                dirtyCrops.append(cropDetection(image, detection_info))
                            else:
                                damagedCrops.append(cropDetection(image, detection_info))

                            if detection_info["above_threshold"]:
                                confident_detections.append(detection_info)

            except Exception as e:
                print(f"Error running model {model_name}: {e}")
                continue

        def analyze_detections(detections):
            # Initialize flags
            has_damage = False
            has_dirt = False

            for detection in detections:
                if not detection["above_threshold"]:
                    continue

                class_name = detection['class'].lower()

                if any(damage_class in class_name for damage_class in damage_classes):
                    has_damage = True
                if any(dirt_class in class_name for dirt_class in dirt_classes):
                    has_dirt = True

            # Determine verdicts
            damage_verdict = "damaged" if has_damage else "not damaged"
            dirt_verdict = "dirty" if has_dirt else "clean"

            return {
                "verdict": damage_verdict,
                "cleanliness": dirt_verdict,
                "total_detections": len(detections),
                "confident_detections": len([d for d in detections if d["above_threshold"]])
            }

        # Analyze the detections
        analysis = analyze_detections(detections)

        # Generate text output with all detections
        text_output = "YOLO Results:\n"

        # Add analysis results
        text_output += f"\nAnalysis Results:\n"
        text_output += f"Verdict: {analysis['verdict']}\n"
        text_output += f"Cleanliness: {analysis['cleanliness']}\n"

        print(text_output)

        return PredictResult(
            clean=CarCleanliness.Clean if analysis['cleanliness'] == "clean" else CarCleanliness.Dirty,
            damage=CarCondition.Damaged if analysis['verdict'] == "damaged" else CarCondition.Good,
            dirtyImages=dirtyCrops,
            damagedImages=damagedCrops
        )


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

def toBase64(crop: np.ndarray) -> str:
    success, data = cv2.imencode(".jpg", crop)
    return base64.b64encode(data.tobytes()).decode("utf-8")


def cropDetection(image: np.ndarray, detection):
    x1, y1, x2, y2 = map(int, detection['bbox'])
    return image[y1:y2, x1:x2]



@app.route('/api/evaluate', methods=["POST"])
def evaluate():
    if "carImages" not in request.files:
        return jsonify({"error": "No images"}), 400
    files = request.files.getlist("carImages")
    if len(files) == 0 or len(files) > 5:
        return jsonify({"error: too many / too little images"}), 400

    images = []
    for file in files:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": f"Failed to decode image {file.filename}"}), 400
        images.append(img)

    print(f"Recevied {len(images)} images")

    result = predict(images=images)
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
