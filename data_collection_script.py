import tensorflow as tf
import tensorflow_hub as hub
import cv2 as cv
import numpy as np
import math
import pandas as pd
import os
from media_pipe_estimator import detect_and_draw_pose, save_annotated_image
from mediapipe_standardization import standardize_pose_keypoints, apply_pose_alignment
 

def create_annoated_image_folder(output_directory):
    # Create output folder for annotated images
    return os.makedirs(output_directory, exist_ok=True)

# df = pd.DataFrame()

# Function to store annotated images and landmarks to CSV
def prepare_extract_and_store_data_original(data_folder_path, annotated_image_dir, csv_filename):
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
                        # Convert back to dictionary format for DataFrame storage
                        row = {"id": len(df), "label": label_dict[label_id]}
                        for idx, landmark_name in enumerate(land_marks_dict.keys()):
                            x, y, z, visibility = land_marks_dict[landmark_name]
                            row[f"{landmark_name}_x"] = x
                            row[f"{landmark_name}_y"] = y
                            row[f"{landmark_name}_z"] = z
                            row[f"{landmark_name}_visibility"] = visibility
                        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                        df.to_csv(csv_filename, index=False)
                    
            label_id += 1
    return df

def prepare_extract_and_store_data_standardized(data_folder_path, annotated_image_dir, csv_filename):
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
                        keypoints_array = np.array(list(land_marks_dict.values()))

                        # Standardize the keypoints
                        standardized_keypoints = standardize_pose_keypoints(keypoints_array, method='shoulder_centered')

                        # Apply alignment to standardized keypoints
                        aligned_keypoints = apply_pose_alignment(standardized_keypoints)

                        # Convert back to dictionary format for DataFrame storage
                        row = {"id": len(df), "label": label_dict[label_id]}
                        for idx, landmark_name in enumerate(land_marks_dict.keys()):
                            x, y, z, visibility = aligned_keypoints[idx]
                            row[f"{landmark_name}_x"] = x
                            row[f"{landmark_name}_y"] = y
                            row[f"{landmark_name}_z"] = z
                            row[f"{landmark_name}_visibility"] = visibility
                        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                        df.to_csv(csv_filename, index=False)
                    
            label_id += 1
    return df

def update_dataframe(df):
    df_columns = df.columns
    i = 0
    df2=pd.DataFrame()
    for row in df.values:
        landmarks_dict = {}
        for collumn in range(0,len(df_columns)-1):
            landmarks_dict[df_columns[collumn].lower()] = row[collumn]
        landmarks_dict_new = {}
        landmarks_dict_new["id"] = landmarks_dict[f"id"]
        landmarks_dict_new["label"] = landmarks_dict[f"label"]
        include = "NOSE,LEFT_EAR,MOUTH_LEFT,LEFT_SHOULDER,LEFT_HIP,RIGHT_SHOULDER,RIGHT_HIP".lower().split(",")
        bad = False
        for value in include:
            if landmarks_dict[f"{value}_visibility"] < 0.9:
                bad = True
                break
            landmarks_dict_new[f"{value.upper()}_x"] = landmarks_dict[f"{value}_x"]
            landmarks_dict_new[f"{value.upper()}_y"] = landmarks_dict[f"{value}_y"]
            landmarks_dict_new[f"{value.upper()}_z"] = landmarks_dict[f"{value}_z"]
        if not bad:
            df2 = pd.concat([df2, pd.DataFrame([landmarks_dict_new])], ignore_index=True)

        i += 1
    return df2



