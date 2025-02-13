import cv2
import numpy as np
from deepface import DeepFace

class FaceAnalyzer:
    def __init__(self):
        """Initialize face detection and analyzer."""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces(self, image):
        """Detect faces in the given image."""
        if isinstance(image, str):
            image = cv2.imread(image)

        # Convert the image to grayscale once for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        return faces

    def analyze_face(self, face_image):
        """Analyze the given face image using DeepFace."""
        try:
            # DeepFace analyze can accept image directly or file path
            analysis = DeepFace.analyze(face_image, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
            
            # DeepFace returns a list, so we'll handle only the first result if present
            if isinstance(analysis, list):
                analysis = analysis[0]

            result = {
                'age': int(analysis.get('age', 0)),
                'gender': str(analysis.get('gender', '')),
                'dominant_race': str(analysis.get('dominant_race', '')),
                'dominant_emotion': str(analysis.get('dominant_emotion', '')),
                'gender_confidence': float(analysis.get('gender_probability', 0) * 100),
                'race_confidence': float(max(analysis.get('race', {}).values()) * 100),
                'emotion_confidence': float(max(analysis.get('emotion', {}).values()) * 100)
            }
            return result

        except Exception as e:
            print(f"Error analyzing face: {e}")
            return None

    def process_image(self, image):
        """Detect faces and analyze them."""
        faces = self.detect_faces(image)
        results = []

        for (x, y, w, h) in faces:
            # Crop the face region from the image for analysis
            face_image = image[y:y+h, x:x+w]
            
            # Analyze the face using DeepFace
            analysis_result = self.analyze_face(face_image)

            if analysis_result:
                analysis_result['face_coordinates'] = (x, y, w, h)
                results.append(analysis_result)

        return results
