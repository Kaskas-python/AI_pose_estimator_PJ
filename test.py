from data_collection_script import create_annoated_image_folder, prepare_extract_and_store_data
from PostureAnalyzer import PostureAnalyzer
import warnings

# Path to the directory containing images
image_dir = "/Users/arminaskurmauskas/AI_pose_estimator_PJ/dataset_posture"  # Change this to your directory pat
# CSV File to store keypoints
csv_filename = "lumbar_kyphosis_dataset_from_images.csv"
# Initialize CSV File

annotated_image_dir = "annotated_images"

create_annoated_image_folder(annotated_image_dir)


df = prepare_extract_and_store_data(image_dir, annotated_image_dir, csv_filename)

df_columns = df.columns

analyzer = PostureAnalyzer("lumbar_kyphosis_model_optimized.h5")
warnings.filterwarnings('ignore')
i = 0
for row in df.values:
    landmarks_dict = {}
    for collumn in range(2,len(df_columns)-1):
        landmarks_dict[df_columns[collumn].lower()] = row[collumn]
    print(i, analyzer.predict_posture(landmarks_dict))
    i += 1
    

        
