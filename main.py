# main.py
from app import ImageAnalysisApp
import urllib.request
import os

def download_yolo_files():
    base_url = "https://pjreddie.com/media/files/"
    files = {
        "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
        "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    }
    
    for filename, url in files.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded {filename}")

if __name__ == '__main__':
    download_yolo_files()

    app = ImageAnalysisApp()
    app.run()