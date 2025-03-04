import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import pandas as pd
import os

# Load MoveNet Lightning model
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
model = movenet.signatures['serving_default']

# Path to the directory containing images
image_dir = "/Users/arminaskurmauskas/AI_pose_estimator_PJ/datasetas_laikysenai"  # Change this to your directory path

# CSV File to store keypoints
csv_filename = "lumbar_kyphosis_dataset_from_images.csv"

# Initialize CSV File
columns = ["image_filename", "left_shoulder_x", "left_shoulder_y", "left_shoulder_z", 
           "right_shoulder_x", "right_shoulder_y", "right_shoulder_z",
           "left_hip_x", "left_hip_y", "left_hip_z", 
           "right_hip_x", "right_hip_y", "right_hip_z", 
           "left_knee_x", "left_knee_y", "left_knee_z", 
           "right_knee_x", "right_knee_y", "right_knee_z"]
df = pd.DataFrame(columns=columns)

# Function to detect keypoints
def detect_keypoints(image):
    img = tf.image.resize_with_pad(tf.convert_to_tensor(image), 192, 192)
    img = tf.cast(img, dtype=tf.int32)
    input_tensor = tf.expand_dims(img, axis=0)

    # Run MoveNet model
    outputs = model(input_tensor)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]

    return keypoints  # Shape: (17, 3) - (x, y, confidence)

# Function to calculate lumbar angle
# def calculate_lumbar_angle(keypoints):
#     def angle_between(p1, p2, p3):
#         a, b, c = np.array(p1), np.array(p2), np.array(p3)
#         ba, bc = a - b, c - b
#         cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#         return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

#     # Extract shoulder, hip, knee keypoints
#     left_shoulder, left_hip, left_knee = keypoints[5][:2], keypoints[11][:2], keypoints[13][:2]
#     right_shoulder, right_hip, right_knee = keypoints[6][:2], keypoints[12][:2], keypoints[14][:2]

#     # Calculate lumbar curvature angles
#     left_angle = angle_between(left_shoulder, left_hip, left_knee)
#     right_angle = angle_between(right_shoulder, right_hip, right_knee)

#     return (left_angle + right_angle) / 2  # Average lumbar angle

# Function to classify posture based on lumbar angle
# def classify_lumbar_angle(lumbar_angle):
#     if lumbar_angle < 45:
#         return "normal"
#     elif 45 <= lumbar_angle <= 55:
#         return "mild_kyphosis"
#     else:
#         return "severe_kyphosis"

# Loop over all images in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Process only .jpg and .png files
        image_path = os.path.join(image_dir, filename)

        # Read image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect keypoints
        keypoints = detect_keypoints(image_rgb)

        # Compute lumbar angle
        # lumbar_angle = calculate_lumbar_angle(keypoints)

        # # Classify posture based on lumbar angle
        # label = classify_lumbar_angle(lumbar_angle) #folder name
    

        # Store keypoints in dataset
        row = [filename] + keypoints[5][:2].tolist() + keypoints[6][:2].tolist() + \
              keypoints[11][:2].tolist() + keypoints[12][:2].tolist() + \
              keypoints[13][:2].tolist() + keypoints[14][:2].tolist() + \
            

        df.loc[len(df)] = row

        # Optionally: display the image and lumbar angle
        cv2.putText(image, f"Lumbar Angle: {lumbar_angle:.2f}°", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, f"Label: {label}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show image
        cv2.imshow("Posture Data Collection", image)
        cv2.waitKey(1000)  # Show each image for 1 second

# Save dataset
df.to_csv(csv_filename, index=False)
print(f"Dataset saved to {csv_filename}")

# Close image window
cv2.destroyAllWindows()
