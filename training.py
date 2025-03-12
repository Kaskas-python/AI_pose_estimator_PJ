from LSTM_model import preprocess_data, run_model
from data_collection_script import create_annoated_image_folder, prepare_extract_and_store_data

# Path to the directory containing images
image_dir = "/Users/arminaskurmauskas/AI_pose_estimator_PJ/dataset_posture"  # Change this to your directory pat
# CSV File to store keypoints
csv_filename = "lumbar_kyphosis_dataset_from_images.csv"
# Initialize CSV File

annotated_image_dir = "annotated_images"

create_annoated_image_folder(annotated_image_dir)


df= prepare_extract_and_store_data(image_dir, annotated_image_dir, csv_filename)

X_train, X_test, y_train, y_test, features = preprocess_data(df)

print("Data preprocessing and splitting complete.")
print("*"* 80)

run_model(X_train, X_test, y_train, y_test, features)

print('Model training and saving complete.')