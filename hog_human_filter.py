import cv2
import numpy as np

# Load pre-trained person detector (HOG + Linear SVM)
def is_human_image(image_path, min_confidence=0.5):
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return False
            
        # Initialize HOG descriptor for people detection
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Detect people
        people, weights = hog.detectMultiScale(
            img, 
            winStride=(8, 8),
            padding=(16, 16), 
            scale=1.05
        )
        
        # If any person detected with confidence above threshold
        if len(weights) > 0 and max(weights) > min_confidence:
            return True
            
        # Fallback - try face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        return len(faces) > 0
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return False