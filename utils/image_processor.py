import cv2

class ImageProcessor:
    @staticmethod
    def load_image(image_path):
        return cv2.imread(image_path)

    @staticmethod
    def convert_to_rgb(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def save_image(image, path):
        return cv2.imwrite(path, image)

    @staticmethod
    def draw_detections(image, detections, faces):
        # Draw detected objects
        for det in detections:
            box = det['box']
            label = f"{det['class']} {det['confidence']:.2f}"
            color = (0, 255, 0)  # Green for objects
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(image, label, (int(box[0]), int(box[1]-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw detected faces
        for face in faces:
            area = face['location']
            x = area['x']
            y = area['y']
            w = area['w']
            h = area['h']
            
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            analysis = face['analysis']
            info_label = f"{analysis['gender']}, {analysis['age']}"
            cv2.putText(image, info_label, (x, y+h+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return image