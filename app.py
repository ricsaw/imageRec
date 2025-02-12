from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import torch
from PIL import Image
from deepface import DeepFace
import matplotlib.pyplot as plt

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_face(face_image_path):
    """Analyze face using DeepFace"""
    try:
        # Analyze face
        analysis = DeepFace.analyze(
            face_image_path, 
            actions=['age', 'gender', 'race', 'emotion'],
            enforce_detection=False
        )
        
        if isinstance(analysis, list):
            analysis = analysis[0]

        # Format results
        result = {
            'age': analysis.get('age'),
            'gender': analysis.get('gender'),
            'dominant_race': analysis.get('dominant_race'),
            'dominant_emotion': analysis.get('dominant_emotion')
        }
        
        return result
    except Exception as e:
        print(f"Face analysis error: {e}")
        return None

def detect_and_identify(image_path):
    """Detect objects and analyze faces in the image"""
    # Load image
    image = cv2.imread(image_path)
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces using DeepFace
    face_identifications = []
    try:
        # Detect faces
        faces = DeepFace.extract_faces(
            image_path,
            enforce_detection=False,
            align=True
        )
        
        for i, face in enumerate(faces):
            # Save face image temporarily
            face_image_path = f"{image_path}_face_{i}.jpg"
            plt.imsave(face_image_path, face['face'])
            
            # Analyze face
            analysis = analyze_face(face_image_path)
            
            if analysis:
                face_identifications.append({
                    'location': face.get('facial_area', {}),
                    'analysis': analysis
                })
            
            # Clean up face image
            os.remove(face_image_path)
            
    except Exception as e:
        print(f"Face detection error: {e}")
    
    # YOLOv5 object detection
    img = Image.open(image_path)
    results = model(img)
    
    # Process YOLOv5 results
    detections = []
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        detections.append({
            'class': results.names[int(cls)],
            'confidence': float(conf),
            'box': [float(x) for x in box]
        })
    
    # Draw faces and objects on the image
    img_cv = cv2.imread(image_path)
    
    # Draw detected objects
    for det in detections:
        box = det['box']
        label = f"{det['class']} {det['confidence']:.2f}"
        color = (0, 255, 0)  # Green for objects
        cv2.rectangle(img_cv, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(img_cv, label, (int(box[0]), int(box[1]-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw detected faces with analysis
    for face in face_identifications:
        area = face['location']
        x = area.get('x', 0)
        y = area.get('y', 0)
        w = area.get('w', 0)
        h = area.get('h', 0)
        
        # Draw rectangle around face
        cv2.rectangle(img_cv, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Draw analysis results
        analysis = face['analysis']
        label = f"Age: {analysis['age']}, {analysis['gender']}"
        cv2.putText(img_cv, label, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw emotion and race on second line
        label2 = f"{analysis['dominant_emotion']}, {analysis['dominant_race']}"
        cv2.putText(img_cv, label2, (x, y-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Save annotated image
    output_path = image_path.replace('.', '_detected.')
    cv2.imwrite(output_path, img_cv)
    
    return detections, face_identifications, output_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Detect objects and analyze faces
            detections, faces, output_path = detect_and_identify(filepath)
            
            # Get relative path for the output image
            relative_path = os.path.relpath(output_path, 'static')
            
            # Clean up original file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'detections': detections,
                'faces': faces,
                'image_path': relative_path
            })
            
        except Exception as e:
            # Clean up if there's an error
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)