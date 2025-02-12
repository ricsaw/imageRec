import os
from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename
from services.detection_services import DetectionService
from config import Config
import cv2
from threading import Thread
import time

class VideoStream:
    def __init__(self, detection_service):
        self.video = cv2.VideoCapture(0)
        self.detection_service = detection_service
        self.is_running = False
        self._current_frame = None
        self._processed_frame = None

    def start(self):
        self.is_running = True
        Thread(target=self._capture_loop).start()

    def stop(self):
        self.is_running = False
        self.video.release()

    def _capture_loop(self):
        while self.is_running:
            success, frame = self.video.read()
            if success:
                self._current_frame = frame
                # Process frame
                result = self.detection_service.process_image(frame)
                self._processed_frame = result.get('annotated_frame', frame)

    def get_frame(self):
        if self._processed_frame is not None:
            ret, jpeg = cv2.imencode('.jpg', self._processed_frame)
            return jpeg.tobytes()
        return None

class ImageAnalysisApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
        self.detection_service = DetectionService()
        self.video_stream = None

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

    def generate_frames(self):
        while True:
            if self.video_stream:
                frame = self.video_stream.get_frame()
                if frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)

    def setup_routes(self):
        @self.app.route('/')
        def home():
            return render_template('index.html')

        @self.app.route('/predict', methods=['POST'])
        def predict():
            return self.handle_prediction()

        @self.app.route('/video')
        def video():
            return render_template('video.html')

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/start_stream', methods=['POST'])
        def start_stream():
            if not self.video_stream:
                self.video_stream = VideoStream(self.detection_service)
                self.video_stream.start()
            return jsonify({'status': 'success'})

        @self.app.route('/stop_stream', methods=['POST'])
        def stop_stream():
            if self.video_stream:
                self.video_stream.stop()
                self.video_stream = None
            return jsonify({'status': 'success'})

    def handle_prediction(self):
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if file and self.allowed_file(file.filename):
            try:
                os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
                filename = secure_filename(file.filename)
                filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                result = self.detection_service.process_image(filepath)
                os.remove(filepath)
                
                if result.get('success', False):
                    return jsonify(result)
                else:
                    return jsonify({'error': result.get('error', 'Unknown error')})
                
            except Exception as e:
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

    def __del__(self):
        if self.video_stream:
            self.video_stream.stop()