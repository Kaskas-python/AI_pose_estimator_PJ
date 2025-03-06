import tensorflow as tf
import tensorflow_hub as hub
import cv2 as cv
import numpy as np
import pandas as pd
import os
from media_pipe_estimator import detect_and_draw_pose, save_annotated_image


# Path to the directory containing images
image_dir = "/Users/arminaskurmauskas/AI_pose_estimator_PJ/dataset_posture"  # Change this to your directory pat
# CSV File to store keypoints
csv_filename = "lumbar_kyphosis_dataset_from_images.csv"
# Initialize CSV File

annotated_image_dir = "annotated_images"

# Create output folder for annotated images
os.makedirs(annotated_image_dir, exist_ok=True)

# # Initialize CSV file columns
# columns = ["id", "label",
#            "left_shoulder_x", "left_shoulder_y", "left_shoulder_z", 
#            "right_shoulder_x", "right_shoulder_y", "right_shoulder_z",
#            "left_hip_x", "left_hip_y", "left_hip_z", 
#            "right_hip_x", "right_hip_y", "right_hip_z", 
#            "left_knee_x", "left_knee_y", "left_knee_z", 
#            "right_knee_x", "right_knee_y", "right_knee_z"]

df = pd.DataFrame()

# Function to store annotated images and landmarks to CSV
def prepare_extract_and_store_data(data_folder_path, df):
    # list subdirectories( for subjects) in the dataset
    dirs= os.listdir(data_folder_path)
    # photos = []
    labels = []
    label_dict = {}
    label_id = 0

    for dir_name in dirs:
        subject_dir_path = os.path.join(data_folder_path, dir_name)
        if not os.path.isdir(subject_dir_path):
            continue

        # Save the mapping of label to condition
        label_dict[label_id] = dir_name

        for image_name in os.listdir(subject_dir_path):
            if image_name.endswith(".jpg") or image_name.endswith(".png"):
                image_path = os.path.join(subject_dir_path, image_name)
                image = cv.imread(image_path)
                if image is None:
                    print(f"Error loading image: {image_path}")
                    continue
                
                # Converts image to gray ,detect keypoints and annotate image
                annotadet_image, land_marks_dict = detect_and_draw_pose(
                    image, visibility_threshold=0.5)
                if annotadet_image is not None:
                    annotadet_image_path = os.path.join(annotated_image_dir, f"{label_id}_{image_name}")
                    save_annotated_image(annotadet_image, annotadet_image_path)
                    labels.append(label_id)
                if land_marks_dict is not None:
                    row = {
                        "id": len(df),
                        "label": label_dict[label_id]
                    }
                    for landmark_name, (x, y, z, visibility) in land_marks_dict.items():
                        row[landmark_name + "_x"] = x
                        row[landmark_name + "_y"] = y
                        row[landmark_name + "_z"] = z
                        row[landmark_name + "_visibility"] = visibility
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                
        label_id += 1
    return df
            
# for filename in os.listdir(image_dir):
#     if filename.endswith(".jpg") or filename.endswith(".png"):  # Process only .jpg and .png files
#         image_path = os.path.join(image_dir, filename)

#         # Read image
#         image = cv2.imread(image_path)
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Detect keypoints
#         keypoints = detect_keypoints(image_rgb)

#         # Compute lumbar angle
#         # lumbar_angle = calculate_lumbar_angle(keypoints)

#         # # Classify posture based on lumbar angle
#         # label = classify_lumbar_angle(lumbar_angle) #folder name
    

#         # Store keypoints in dataset
#         row = [filename] + keypoints[5][:2].tolist() + keypoints[6][:2].tolist() + \
#               keypoints[11][:2].tolist() + keypoints[12][:2].tolist() + \
#               keypoints[13][:2].tolist() + keypoints[14][:2].tolist() + \
            

#         df.loc[len(df)] = row

#         # Optionally: display the image and lumbar angle
#         cv2.putText(image, f"Lumbar Angle: {lumbar_angle:.2f}°", (20, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#         cv2.putText(image, f"Label: {label}", (20, 80),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#         # Show image
#         cv2.imshow("Posture Data Collection", image)
#         cv2.waitKey(1000)  # Show each image for 1 second

df = prepare_extract_and_store_data(image_dir, df)
# Save dataset
df.to_csv(csv_filename, index=False)
print(f"Dataset saved to {csv_filename}")

# # Close image window
# cv2.destroyAllWindows()
