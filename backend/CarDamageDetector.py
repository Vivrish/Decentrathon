from ultralytics import YOLO


class CarDamageDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        """
        Initialize YOLOv8 detector

        Args:
            model_path: Path to the trained YOLOv8 model (best.pt)
            conf_threshold: Confidence threshold for detection
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        # Define class names (adjust these based on your model's classes)
        self.class_names = {
            0: 'dent',
            1: 'scratch'
            # Add more classes if your model has them
        }

    def detect_from_image(self, image_path):
        """
        Detect objects in an image using YOLOv8

        Args:
            image_path: Path to the input image

        Returns:
            List of detected objects with their classes and confidence
        """
        # Perform detection
        results = self.model.predict(image_path, conf=self.conf_threshold, verbose=False)

        # Parse results
        detections = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.class_names.get(cls, f'class_{cls}')
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': box.xyxy[0].tolist() if hasattr(box, 'xyxy') else []
                })
                print(f"Detection! {detections[len(detections) - 1]}")

        if len(detections) == 0:
            print("No detection")

        return detections