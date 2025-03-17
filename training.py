from Simple_Neural_Network import preprocess_data, run_model
from data_collection_script import create_annoated_image_folder, prepare_extract_and_store_data_original, prepare_extract_and_store_data_original, update_dataframe
from data_manipulation import augment_keypoint_data

# Path to the directory containing images
image_dir = "/Users/arminaskurmauskas/AI_pose_estimator_PJ/dataset_posture"
# CSV File to store keypoints
csv_filename = "lumbar_kyphosis_dataset_from_images.csv"
# Initialize CSV File

annotated_image_dir = "annotated_images"

create_annoated_image_folder(annotated_image_dir)


df = prepare_extract_and_store_data_original(image_dir, annotated_image_dir, csv_filename)
df = update_dataframe(df)

df_augmented = augment_keypoint_data(df)

print (df_augmented.info())
X_train, X_test, y_train, y_test, features = preprocess_data(df_augmented)

print("Data preprocessing and splitting complete.")
print("*"* 80)

run_model(X_train, X_test, y_train, y_test, features)

print('Model training and saving complete.')