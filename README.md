# imageRec


# Image Analysis Application

A web application that performs real-time face analysis and object detection on uploaded images. The application uses deep learning models to detect faces, analyze facial attributes (age, gender, emotion, race), and identify objects in images.

## Features

* 🎯 **Object Detection using YOLOv3**
  * Detects 80+ different object classes
  * Real-time bounding box visualization
  * Confidence scores for detections
* 👤 **Face Analysis using DeepFace**
  * Age estimation
  * Gender detection
  * Emotion recognition
  * Race classification
  * Confidence scores for each attribute
* 🖼️ **Image Processing**
  * Supports multiple image formats (JPG, PNG, JPEG, WEBP)
  * Automatic image enhancement
  * Visual feedback with bounding boxes
  * Labeled detections
* 💫 **Interactive UI**
  * Drag and drop file upload
  * Real-time processing feedback
  * Clean and intuitive interface
  * Mobile responsive design

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/imageRec.git
cd imageRec
```

2. Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python main.py
```

The application will be available at `http://localhost:5000`

## Project Structure

```
imageRec/
├── static/
│   └── uploads/           # Temporary image storage
├── templates/
│   └── index.html         # Main web interface
├── models/               # Pre-trained models (YOLOv3, DeepFace)
|   └── face_analyzer.py  # Face analysis model wrapper
|   └── object_detector.py# Object detection model wrapper
├── services/           # Business logic and processing functions
│   └── detection_service.py # Object detection service
├── utils/               # Helper functions and utilities
|   └── image_processor.py # Image processing utilities
├── main.py              # Flask application entry point
├── app.py               # Flask application configuration and routes
├── requirements.txt     # Project dependencies



```

## Dependencies

```txt
flask==2.0.1
opencv-python-headless==4.5.3.56
deepface==0.0.75
numpy==1.21.0
tensorflow-cpu>=2.5.0
```

## Usage

1. Open the application in a web browser
2. Upload an image by:
   * Clicking the upload area
   * Dragging and dropping a file
3. Wait for processing to complete
4. View results showing:
   * Detected objects with confidence scores
   * Face analysis results
   * Annotated image with detections

## API Endpoints

### **POST /predict**

Analyzes an uploaded image for faces and objects.

#### **Request:**

* **Method:** POST
* **Content-Type:** multipart/form-data
* **Body:** file (image file)

#### **Response:**

```json
{
  "success": true,
  "detections": [
    {
      "class": "person",
      "confidence": 0.95,
      "box": [x, y, width, height]
    }
  ],
  "faces": [
    {
      "location": {
        "x": 100,
        "y": 100,
        "w": 100,
        "h": 100
      },
      "analysis": {
        "age": 25,
        "gender": "Man",
        "dominant_emotion": "happy",
        "dominant_race": "asian",
        "gender_confidence": 99.9,
        "emotion_confidence": 95.5,
        "race_confidence": 90.0
      }
    }
  ],
  "image_path": "path_to_processed_image"
}
```

## Features in Detail

### **Object Detection**

* Uses YOLOv3 model
* Detects common objects like:
  * People
  * Animals
  * Vehicles
  * Furniture
  * Electronics
  * And more...

### **Face Analysis**

#### **Age Estimation**

* Numerical age prediction
* Age range classification

#### **Gender Detection**

* Binary classification (Man/Woman)
* Confidence score

#### **Emotion Recognition**

* 7 basic emotions:
  * Happy
  * Sad
  * Angry
  * Surprised
  * Fearful
  * Disgusted
  * Neutral

#### **Race Classification**

* Multiple racial categories
* Confidence scores
* Ethical consideration notice

## Technical Implementation

### **Backend**

* Flask web framework
* OpenCV for image processing
* DeepFace for facial analysis
* YOLOv3 for object detection
* NumPy for numerical operations

### **Frontend**

* HTML5 for structure
* CSS3 for styling
* JavaScript for interactivity
* Bootstrap 5 for responsive design
* Font Awesome for icons

## Performance Considerations

* **Image Size:** Optimal results with images under 2048x2048 pixels
* **Processing Time:** 2-5 seconds per image typically
* **Memory Usage:** ~500MB RAM during processing
* **Storage:** Temporary files cleaned up after processing

## Error Handling

The application handles various error cases:

* Invalid file types
* File size limits
* Processing failures
* Network issues
* Server errors

## Security Features

* Secure file handling
* Input validation
* File type checking
* Size limitations
* Temporary file cleanup
* Error logging

## Known Limitations

* Processing time varies with image size
* Face detection accuracy depends on image quality
* Some objects may be misclassified
* Memory usage increases with image size
