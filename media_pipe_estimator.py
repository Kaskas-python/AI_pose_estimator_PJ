import cv2 as cv
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os

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
            x= int(landmark.x *image.shape[1])
            y= int(landmark.y * image.shape[0])
            z= landmark.z
            visibility = landmark.visibility

            # Add to dict
            landmark_dict[landmark_name]= (x, y, z, visibility)

            # Draw on image only if visibility is above the threshold
            if visibility > visibility_threshold:
                # Draw point
                cv.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

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
    

import math

def calculate_lumbar_angle(landmarks):
    """
    Calculate the lumbar kyphosis angle using 3D pose landmarks.
    """
    try:
        left_shoulder = landmarks['LEFT_SHOULDER']
        right_shoulder = landmarks['RIGHT_SHOULDER']
        left_hip = landmarks['LEFT_HIP']
        right_hip = landmarks['RIGHT_HIP']

        # Compute midpoints (average of left and right)
        shoulder_mid = (
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2,
            (left_shoulder[2] + right_shoulder[2]) / 2
        )
        hip_mid = (
            (left_hip[0] + right_hip[0]) / 2,
            (left_hip[1] + right_hip[1]) / 2,
            (left_hip[2] + right_hip[2]) / 2
        )

        # Calculate differences
        delta_y = shoulder_mid[1] - hip_mid[1]  # Vertical distance
        delta_z = shoulder_mid[2] - hip_mid[2]  # Depth distance

        # Compute lumbar angle
        lumbar_angle = math.degrees(math.atan2(delta_z, delta_y))

        return lumbar_angle

    except KeyError as e:
        print(f"Missing keypoint: {e}")
        return None


# image_path = '/Users/arminaskurmauskas/Pose_Estimator/input_image.jpeg'
image = cv.imread(image_path)

annotated_image, pose_landmarks_dict = detect_and_draw_pose(
    image, visibility_threshold=0.6)

if pose_landmarks_dict:
    for name, (x, y, z, visibility) in pose_landmarks_dict.items():
        print(f'{name}: ({x}, {y}), Z- {z}, visibility: {visibility:.2f}')


plt.figure(figsize=(8, 6))
plt.imshow(cv.cvtColor(annotated_image, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# lumbar_angle = calculate_lumbar_angle(pose_landmarks_dict)

# if lumbar_angle is not None:
#     print(f"Lumbar Kyphosis Angle: {lumbar_angle:.2f} degrees")
# else:
#     print("Error calculating lumbar angle.")

output_path = image_path+'_annotated_image.jpg'
cv.imwrite(output_path, annotated_image)

# print(get_landmarks_from_image(image))