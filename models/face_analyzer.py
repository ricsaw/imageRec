import cv2
import numpy as np
from deepface import DeepFace

class FaceAnalyzer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces(self, image):
        """Detect faces in image"""
        if isinstance(image, str):
            image = cv2.imread(str(image))
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces

    def analyze_face(self, face_image):
        """Analyze face using DeepFace"""
        try:
            analysis = DeepFace.analyze(
                str(face_image),
                actions=['age', 'gender', 'race', 'emotion'],
                enforce_detection=False
            )
            
            if isinstance(analysis, list):
                analysis = analysis[0]

            # Convert numpy values to Python native types
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
            print(f"Face analysis error: {e}")
            return None