import cv2
import os
from PostureAnalyzer import PostureAnalyzer

input_directory ='/Users/arminaskurmauskas/AI_pose_estimator_PJ/test_images'
output_directory ='test_annotations'

os.makedirs(output_directory, exist_ok=True)

lst= os.listdir(output_directory)
number_of_files = len(lst)

model_path= "lumbar_kyphosis_model_with_scaler.h5"

def analyze_image(image, model_path):
    analyzer= PostureAnalyzer(model_path)

    annotated_frame, landmark_dict, success = analyzer.extract_landmarks(image)
    if success:

        prediction= analyzer.predict_posture(landmark_dict)

        text = f"Prediction: {prediction}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (00, 30)
        fontScale = 0.3
        color = (0, 0, 255)
        thickness = 1
        cv2.putText(
            annotated_frame, text, org, font, fontScale,
            color, thickness, cv2.LINE_AA
                )

    return annotated_frame, prediction

def save_annotated_image(annotated_frame, output_path):
    cv2.imwrite(output_path, annotated_frame)
    print(f"Annotated frame saved as {output_path}")



for i,filename in enumerate(os.listdir(input_directory), start= number_of_files +1):
    number_of_files = len(lst)
    
    file_path= os.path.join(input_directory, filename)
    image = cv2.imread(file_path)

    annotated_frame, prediction= analyze_image(image, model_path)

    output_path= os.path.join(output_directory, f"annotated_image_{i}.jpg")

    save_annotated_image(annotated_frame, output_path)

    print(f"Predicted posture to image {i}: {prediction}")