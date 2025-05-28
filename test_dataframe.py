from data_collection_script import create_annoated_image_folder, prepare_extract_and_store_data_original
from PostureAnalyzer import PostureAnalyzer
import warnings
import pandas as pd

# Path to the directory containing images
image_dir = "/Users/arminaskurmauskas/AI_pose_estimator_PJ/dataset_posture"  # Change this to your directory pat
# CSV File to store keypoints
csv_filename = "lumbar_kyphosis_dataset_from_images.csv"
# Initialize CSV File

annotated_image_dir = "annotated_images"

create_annoated_image_folder(annotated_image_dir)


df = prepare_extract_and_store_data_original(image_dir, annotated_image_dir, csv_filename)

df_columns = df.columns

analyzer = PostureAnalyzer("lumbar_kyphosis_model_with_scaler.h5")
warnings.filterwarnings('ignore')
i = 0
df2=pd.DataFrame()
for row in df.values:
    landmarks_dict = {}
    for collumn in range(0,len(df_columns)-1):
        landmarks_dict[df_columns[collumn].lower()] = row[collumn]
    landmarks_dict_new = {}
    landmarks_dict_new["id"] = landmarks_dict[f"id"]
    landmarks_dict_new["label"] = landmarks_dict[f"label"]
    include = "NOSE,LEFT_EAR,MOUTH_LEFT,LEFT_SHOULDER,LEFT_HIP,RIGHT_SHOULDER,right_hip".lower().split(",")
    bad = False
    for value in include:
        if landmarks_dict[f"{value}_visibility"] < 0.5:
            bad = True
            break
        landmarks_dict_new[f"{value.upper()}_x"] = landmarks_dict[f"{value}_x"]
        landmarks_dict_new[f"{value.upper()}_y"] = landmarks_dict[f"{value}_y"]
        landmarks_dict_new[f"{value.upper()}_z"] = landmarks_dict[f"{value}_z"]
    if not bad:
        df2 = pd.concat([df2, pd.DataFrame([landmarks_dict_new])], ignore_index=True)
        # print(landmarks_dict_new)
    print(i, analyzer.predict_posture(landmarks_dict))
    i += 1
df2.to_csv("lumbar_kyphosis_dataset_from_images_updated.csv", index=False)

    

        
