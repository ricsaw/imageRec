import cv2
import os
import numpy as np
from models.face_analyzer import FaceAnalyzer
from models.object_detector import ObjectDetector

class DetectionService:
    def __init__(self):
        self.face_analyzer = FaceAnalyzer()
        self.object_detector = ObjectDetector()

        # Initialize YOLO with CUDA support if available
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def convert_to_serializable(self, obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        return obj

    def process_image(self, image_path):
        try:
            # Convert image_path to string
            image_path = str(image_path)
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")

            # Make a copy for drawing
            output_image = image.copy()

            # Detect and analyze faces in parallel (using threading or multiprocessing)
            faces = self.face_analyzer.detect_faces(image)
            face_analyses = []

            for (x, y, w, h) in faces:
                face_img = image[y:y+h, x:x+w]
                analysis = self.face_analyzer.analyze_face(face_img)

                if analysis:
                    face_analyses.append({
                        'location': {
                            'x': int(x),
                            'y': int(y),
                            'w': int(w),
                            'h': int(h)
                        },
                        'analysis': self.convert_to_serializable(analysis)
                    })

                    # Draw face rectangle and info
                    cv2.rectangle(output_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    text = f"{analysis['gender']}, {analysis['age']}"
                    cv2.putText(output_image, text, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Detect objects
            detections = self.object_detector.detect_objects(image)
            # Convert detections to serializable format
            detections = self.convert_to_serializable(detections)

            # Draw object detections
            for det in detections:
                box = det['box']
                label = f"{det['class']} {det['confidence']:.2f}"
                cv2.rectangle(output_image, 
                            (int(box[0]), int(box[1])), 
                            (int(box[2]), int(box[3])), 
                            (0, 255, 0), 2)
                cv2.putText(output_image, label, 
                          (int(box[0]), int(box[1]-10)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save output image
            output_path = image_path.replace('.', '_analyzed.')
            cv2.imwrite(output_path, output_image)

            return {
                'success': True,
                'detections': detections,
                'faces': face_analyses,
                'image_path': os.path.relpath(output_path, 'static')
            }

        except Exception as e:
            print(f"Error processing image: {e}")
            return {
                'success': False,
                'error': str(e)
            }
