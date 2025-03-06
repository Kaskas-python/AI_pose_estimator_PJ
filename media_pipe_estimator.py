import cv2 as cv
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Get original MediaPipe landmarks from an image

def get_landmarks_from_image(image):
    #Load MediaPipe model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Convert image to RGB
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Perform estimation

    result = pose.process(image_rgb)

    if result.pose_landmarks:

        landmark_dict = {}

        for i, landmark in enumerate(result.pose_landmarks.landmark):
            landmark_name= mp_pose.PoseLandmark(i).name
            x= landmark.x
            y= landmark.y
            z= landmark.z
            visibly= landmark.visibility

            # Add values to dict
            landmark_dict[landmark_name] = (x, y, z, visibly)

    return landmark_dict

def save_annotated_image(annotated_image, output_image_path):
    cv.imwrite(output_image_path, annotated_image)
    print(f"Image saved as {output_image_path}")
    

"""Function that detects poses and draw them in the image"""

def detect_and_draw_pose(image, visibility_threshold=0.5):
    # Load MediaPipe model
    mp_pose= mp.solutions.pose
    pose= mp_pose.Pose()

    # Convert image to RGB
    image_rgb= cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Estimation
    results= pose.process(image_rgb)

    # Draw poses
    if results.pose_landmarks:
        mp.drawing= mp.solutions.drawing_utils
        annotated_image= image.copy()

        landmark_dict= {}
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            landmark_name = mp_pose.PoseLandmark(i).name
            x= landmark.x
            x_d= int(x *image.shape[1])
            y= landmark.y
            y_d= int(y * image.shape[0])
            z= landmark.z
            visibility = landmark.visibility

            # Add to dict
            landmark_dict[landmark_name]= (x, y, z, visibility)

            # Draw on image only if visibility is above the threshold
            if visibility > visibility_threshold:
                # Draw point
                cv.circle(annotated_image, (x_d, y_d), 5, (0, 255, 0), -1)

        # Draw connections
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            if results.pose_landmarks.landmark[start_idx].visibility > visibility_threshold and results.pose_landmarks.landmark[end_idx].visibility > visibility_threshold:
                start_point = (int(results.pose_landmarks.landmark[start_idx].x * image.shape[1]), 
                               int(results.pose_landmarks.landmark[start_idx].y * image.shape[0]))
                end_point = (int(results.pose_landmarks.landmark[end_idx].x * image.shape[1]), 
                             int(results.pose_landmarks.landmark[end_idx].y * image.shape[0]))
                cv.line(annotated_image, start_point, end_point, (0, 255, 0), 2)

        return annotated_image, landmark_dict
    else:
        return image, None
    