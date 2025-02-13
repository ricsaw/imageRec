import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, weights_path="yolov3.weights", config_path="yolov3.cfg", classes_path="coco.names"):
        """
        Initializes the ObjectDetector with the YOLO model and class labels.
        :param weights_path: Path to YOLO model weights file.
        :param config_path: Path to YOLO model config file.
        :param classes_path: Path to file containing class labels.
        """
        # Load YOLO model
        self.net = cv2.dnn.readNet(weights_path, config_path)

        # Load class labels (COCO dataset labels)
        try:
            with open(classes_path, "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            raise Exception(f"Class labels file '{classes_path}' not found.")

        # Get YOLO output layers
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, image_path):
        """
        Detect objects in the given image.
        :param image_path: Path to image file or a NumPy array representing an image.
        :return: A list of detections with class name, confidence, and bounding box.
        """
        # Check if image_path is a path to an image or an image array
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Image at path '{image_path}' could not be loaded.")
        else:
            image = image_path

        # Get image dimensions
        height, width = image.shape[:2]

        # Preprocess image (prepare for YOLO)
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)

        # Run forward pass to get detections
        outs = self.net.forward(self.output_layers)

        # Initialize lists to hold detection results
        detections = []

        # Loop through detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)  # Get class with highest confidence score
                confidence = float(scores[class_id])

                # Only consider detections with confidence > 0.5
                if confidence > 0.5:
                    # Calculate bounding box coordinates (scaled back to original image size)
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Add detection to list
                    detections.append({
                        'class': self.classes[class_id],
                        'confidence': confidence,
                        'box': [x, y, x + w, y + h]
                    })

        return detections

    def draw_detections(self, image, detections):
        """
        Draw bounding boxes and labels on the image.
        :param image: The image to draw on.
        :param detections: A list of detections to draw.
        :return: Image with drawn bounding boxes and labels.
        """
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['box'])
            label = f"{detection['class']} ({detection['confidence']:.2f})"
            color = (0, 255, 0)  # Green color for bounding box

            # Draw rectangle around detected object
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image

