import tensorflow as tf
import tensorflow_hub as hub
import cv2 as cv
import numpy as np
import pandas as pd
import os
from media_pipe_estimator import detect_and_draw_pose, save_annotated_image


# # Path to the directory containing images
# image_dir = "/Users/arminaskurmauskas/AI_pose_estimator_PJ/dataset_posture"  # Change this to your directory pat
# # CSV File to store keypoints
# csv_filename = "lumbar_kyphosis_dataset_from_images.csv"
# # Initialize CSV File

# annotated_image_dir = "annotated_images"

def create_annoated_image_folder(output_directory):
    # Create output folder for annotated images
    return os.makedirs(output_directory, exist_ok=True)

# df = pd.DataFrame()

# Function to store annotated images and landmarks to CSV
def prepare_extract_and_store_data(data_folder_path, annotated_image_dir, csv_filename):
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        print(f"Loaded existing dataset: {csv_filename}")
        return df
    else:
        print("No existing dataset found. Creating a new one.")
    
        # list subdirectories( for subjects) in the dataset
        dirs= os.listdir(data_folder_path)
        labels = []
        label_dict = {}
        label_id = 0
        df = pd.DataFrame()

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
                        df.to_csv(csv_filename, index=False)
                    
            label_id += 1
    return df
            



