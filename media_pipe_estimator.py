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
            y= landmark.y
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
    


# def prepare_data_for_keypoint_extraction(data_folder_path):
#     # list subdirectories( for subjects) in the dataset
#     dirs= os.listdir(data_folder_path)
#     photos = []
#     labels = []
#     label_dict = {}
#     label_id = 0

#     for dir_name in dirs:
#         subject_dir_path = os.path.join(data_folder_path, dir_name)
#         if not os.path.isdir(subject_dir_path):
#             continue

#         # Save the mapping of label to condition
#         label_dict[label_id] = dir_name

#         for image_name in os.listdir(subject_dir_path):
#             image_path = os.path.join(subject_dir_path, image_name)
#             image = cv.imread(image_path)
#             if image is None:
#                 print(f"Error loading image: {image_path}")
#                 continue
#             # Convert image to grayscale
#             gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#             # Detect keypoints and annotate image
#             annotadet_image, land_marks_dict = detect_and_draw_pose(
#                 gray_image, visibility_threshold=0.6)
#             if annotadet_image is not None:
#                 photos.append(annotadet_image)
#                 labels.append(label_id)

#         label_id += 1
            




# image_path = '/Users/arminaskurmauskas/Pose_Estimator/input_image.jpeg'
# image = cv.imread(image_path)

# annotated_image, pose_landmarks_dict = detect_and_draw_pose(
#     image, visibility_threshold=0.6)

# if pose_landmarks_dict:
#     for name, (x, y, z, visibility) in pose_landmarks_dict.items():
#         print(f'{name}: ({x}, {y}), Z- {z}, visibility: {visibility:.2f}')


# plt.figure(figsize=(8, 6))
# plt.imshow(cv.cvtColor(annotated_image, cv.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()


# output_path = image_path+'_annotated_image.jpg'
# cv.imwrite(output_path, annotated_image)

# print(get_landmarks_from_image(image))