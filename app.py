# app.py
import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from services.detection_services import DetectionService
from config import Config

class ImageAnalysisApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
        self.detection_service = DetectionService()

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

    def setup_routes(self):
        @self.app.route('/')
        def home():
            return render_template('index.html')

        @self.app.route('/predict', methods=['POST'])
        def predict():
            return self.handle_prediction()

    def handle_prediction(self):
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if file and self.allowed_file(file.filename):
            try:
                # Create uploads directory if it doesn't exist
                os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
                
                # Save the uploaded file
                filename = secure_filename(file.filename)
                filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Process the image
                result = self.detection_service.process_image(filepath)
                
                # Clean up original file
                os.remove(filepath)
                
                if result.get('success', False):
                    return jsonify(result)
                else:
                    return jsonify({'error': result.get('error', 'Unknown error')})
                
            except Exception as e:
                # Clean up if there's an error
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except:
                        pass
                return jsonify({'error': str(e)})
        
        return jsonify({'error': 'Invalid file type'})

    def run(self):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        self.setup_routes()
        self.app.run(debug=True)