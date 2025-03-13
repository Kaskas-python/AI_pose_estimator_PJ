import tensorflow as tf
import tensorflow_hub as hub
import cv2 as cv
import numpy as np
import pandas as pd
import os
from media_pipe_estimator import detect_and_draw_pose, save_annotated_image


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

import numpy as np
import pandas as pd

def extract_anatomical_features(df):
    """
    Extract derived features that capture anatomical relationships between spinal keypoints,
    incorporating visibility scores to handle potentially occluded or unreliable keypoints.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing keypoint coordinates and visibility scores
                          (assuming columns like 'x1', 'y1', 'z1', 'v1', 'x2', 'y2', etc.)
                          where 'v' represents visibility/confidence score
    
    Returns:
    pandas.DataFrame: Original DataFrame with additional anatomical features
    """
    # Create a copy to avoid modifying the original DataFrame
    enhanced_df = df.copy()
    
    # Detect the number of keypoints and whether we have 3D data
    has_z = any('z' in col for col in df.columns)
    num_keypoints = 0
    
    # Determine how many keypoints we have
    for i in range(1, 100):  # Upper limit of 100 keypoints
        if f'x{i}' in df.columns:
            num_keypoints = i
        else:
            break
    
    print(f"Detected {num_keypoints} keypoints in the dataset")
    print(f"Using {'3D' if has_z else '2D'} coordinate data with visibility scores")
    
    # Define a visibility threshold for reliable keypoints
    VISIBILITY_THRESHOLD = 0.7  # Adjust based on your data's visibility score range
    
    # 1. Create a reliability mask for calculations
    # We'll use this to handle missing or unreliable keypoints
    reliable_keypoints = {}
    for i in range(1, num_keypoints+1):
        if f'v{i}' in df.columns:
            reliable_keypoints[i] = df[f'v{i}'] >= VISIBILITY_THRESHOLD
        else:
            # If visibility score not available, assume keypoint is reliable
            reliable_keypoints[i] = pd.Series([True] * len(df))
    
    # 2. Spinal Curvature Features (Angles)
    
    # Calculate angles between consecutive vertebrae
    for i in range(1, num_keypoints-1):
        # For each set of three consecutive points, calculate the angle if all points are reliable
        # Create a mask where all three points are reliable
        reliable_triplet = (reliable_keypoints[i] & 
                           reliable_keypoints[i+1] & 
                           reliable_keypoints[i+2])
        
        if has_z:
            # Extract 3D coordinates for three consecutive points
            p1 = df[[f'x{i}', f'y{i}', f'z{i}']].values
            p2 = df[[f'x{i+1}', f'y{i+1}', f'z{i+1}']].values
            p3 = df[[f'x{i+2}', f'y{i+2}', f'z{i+2}']].values
            
            # Calculate vectors between points
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Calculate angles between vectors (in degrees)
            angles = np.full(len(df), np.nan)  # Initialize with NaN
            
            # Only calculate for reliable triplets
            for j in range(len(df)):
                if reliable_triplet.iloc[j]:
                    dot_product = np.dot(v1[j], v2[j])
                    norm_v1 = np.linalg.norm(v1[j])
                    norm_v2 = np.linalg.norm(v2[j])
                    
                    # Handle potential numerical issues
                    cos_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
                    angle = np.arccos(cos_angle) * 180 / np.pi
                    angles[j] = angle
            
            enhanced_df[f'angle_{i}_{i+1}_{i+2}'] = angles
        else:
            # Extract 2D coordinates for three consecutive points
            p1 = df[[f'x{i}', f'y{i}']].values
            p2 = df[[f'x{i+1}', f'y{i+1}']].values
            p3 = df[[f'x{i+2}', f'y{i+2}']].values
            
            # Calculate vectors between points
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Calculate angles between vectors (in degrees)
            angles = np.full(len(df), np.nan)  # Initialize with NaN
            
            # Only calculate for reliable triplets
            for j in range(len(df)):
                if reliable_triplet.iloc[j]:
                    dot_product = np.dot(v1[j], v2[j])
                    norm_v1 = np.linalg.norm(v1[j])
                    norm_v2 = np.linalg.norm(v2[j])
                    
                    # Handle potential numerical issues
                    cos_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
                    angle = np.arccos(cos_angle) * 180 / np.pi
                    angles[j] = angle
            
            enhanced_df[f'angle_{i}_{i+1}_{i+2}'] = angles
        
        # Add a weighted angle feature that takes visibility into account
        if f'v{i}' in df.columns and f'v{i+1}' in df.columns and f'v{i+2}' in df.columns:
            visibility_weights = (df[f'v{i}'] + df[f'v{i+1}'] + df[f'v{i+2}']) / 3
            enhanced_df[f'weighted_angle_{i}_{i+1}_{i+2}'] = enhanced_df[f'angle_{i}_{i+1}_{i+2}'] * visibility_weights
    
    # 3. Vertebral Distances
    
    # Calculate distances between consecutive vertebrae
    for i in range(1, num_keypoints):
        # Create a mask where both points are reliable
        reliable_pair = reliable_keypoints[i] & reliable_keypoints[i+1]
        
        if has_z:
            # Extract 3D coordinates for two consecutive points
            p1 = df[[f'x{i}', f'y{i}', f'z{i}']].values
            p2 = df[[f'x{i+1}', f'y{i+1}', f'z{i+1}']].values
            
            # Calculate Euclidean distance between points
            distances = np.full(len(df), np.nan)  # Initialize with NaN
            
            # Only calculate for reliable pairs
            for j in range(len(df)):
                if reliable_pair.iloc[j]:
                    distances[j] = np.sqrt(np.sum((p2[j] - p1[j])**2))
            
            enhanced_df[f'distance_{i}_{i+1}'] = distances
            
            # Calculate horizontal and vertical components of distance (clinically relevant)
            if reliable_pair.any():
                # Horizontal distance (in the transverse plane)
                h_distances = np.full(len(df), np.nan)
                for j in range(len(df)):
                    if reliable_pair.iloc[j]:
                        h_distances[j] = np.sqrt((p2[j][0] - p1[j][0])**2 + (p2[j][2] - p1[j][2])**2)
                enhanced_df[f'horizontal_distance_{i}_{i+1}'] = h_distances
                
                # Vertical distance (along y-axis, assuming y is vertical)
                v_distances = np.full(len(df), np.nan)
                for j in range(len(df)):
                    if reliable_pair.iloc[j]:
                        v_distances[j] = abs(p2[j][1] - p1[j][1])
                enhanced_df[f'vertical_distance_{i}_{i+1}'] = v_distances
        else:
            # Extract 2D coordinates for two consecutive points
            p1 = df[[f'x{i}', f'y{i}']].values
            p2 = df[[f'x{i+1}', f'y{i+1}']].values
            
            # Calculate Euclidean distance between points
            distances = np.full(len(df), np.nan)  # Initialize with NaN
            
            # Only calculate for reliable pairs
            for j in range(len(df)):
                if reliable_pair.iloc[j]:
                    distances[j] = np.sqrt(np.sum((p2[j] - p1[j])**2))
            
            enhanced_df[f'distance_{i}_{i+1}'] = distances
    
    # 4. Spine Alignment Features
    
    # Calculate the overall spinal alignment using reliable keypoints
    if num_keypoints >= 3:
        # Find the most reliable top and bottom keypoints
        reliable_top_kp = 1
        reliable_bottom_kp = num_keypoints
        
        # For mid keypoint, use the middle of the spine
        reliable_mid_kp = num_keypoints // 2
        
        # Create a mask where all three reference points are reliable
        reliable_alignment = (reliable_keypoints[reliable_top_kp] & 
                             reliable_keypoints[reliable_mid_kp] & 
                             reliable_keypoints[reliable_bottom_kp])
        
        if has_z:
            top = df[[f'x{reliable_top_kp}', f'y{reliable_top_kp}', f'z{reliable_top_kp}']].values
            mid = df[[f'x{reliable_mid_kp}', f'y{reliable_mid_kp}', f'z{reliable_mid_kp}']].values
            bottom = df[[f'x{reliable_bottom_kp}', f'y{reliable_bottom_kp}', f'z{reliable_bottom_kp}']].values
            
            # Calculate perpendicular distance from mid-point to the line joining top and bottom
            spine_deviations = np.full(len(df), np.nan)  # Initialize with NaN
            
            for j in range(len(df)):
                if reliable_alignment.iloc[j]:
                    # 3D case - calculate distance from point to line
                    a = top[j]
                    b = bottom[j]
                    p = mid[j]
                    
                    # Vector from a to b
                    d = b - a
                    # Length of the line segment
                    line_length = np.linalg.norm(d)
                    # Normalize direction vector
                    d = d / line_length if line_length > 0 else d
                    
                    # Vector from a to p
                    v = p - a
                    
                    # Projection of v onto d
                    t = np.dot(v, d)
                    
                    # Closest point on the line
                    closest = a + t * d
                    
                    # Distance from mid-point to the line
                    spine_deviations[j] = np.linalg.norm(p - closest)
            
            enhanced_df['spine_deviation'] = spine_deviations
        else:
            top = df[[f'x{reliable_top_kp}', f'y{reliable_top_kp}']].values
            mid = df[[f'x{reliable_mid_kp}', f'y{reliable_mid_kp}']].values
            bottom = df[[f'x{reliable_bottom_kp}', f'y{reliable_bottom_kp}']].values
            
            # Calculate perpendicular distance from mid-point to the line joining top and bottom
            spine_deviations = np.full(len(df), np.nan)  # Initialize with NaN
            
            for j in range(len(df)):
                if reliable_alignment.iloc[j]:
                    a = top[j]
                    b = bottom[j]
                    p = mid[j]
                    
                    # Use the formula for distance from a point to a line
                    num = abs((b[1] - a[1]) * p[0] - (b[0] - a[0]) * p[1] + b[0] * a[1] - b[1] * a[0])
                    denom = np.sqrt((b[1] - a[1])**2 + (b[0] - a[0])**2)
                    
                    if denom > 0:
                        spine_deviations[j] = num / denom
            
            enhanced_df['spine_deviation'] = spine_deviations
        
        # 5. Spine Curvature Ratio
        
        # Calculate the ratio of actual spine length to direct distance (important for kyphosis)
        actual_lengths = np.zeros(len(df))
        segment_counts = np.zeros(len(df))
        
        for i in range(1, num_keypoints):
            # Create a mask where both points are reliable
            reliable_pair = reliable_keypoints[i] & reliable_keypoints[i+1]
            
            if has_z:
                p1 = df[[f'x{i}', f'y{i}', f'z{i}']].values
                p2 = df[[f'x{i+1}', f'y{i+1}', f'z{i+1}']].values
            else:
                p1 = df[[f'x{i}', f'y{i}']].values
                p2 = df[[f'x{i+1}', f'y{i+1}']].values
            
            # Add segment length to total only for reliable segments
            for j in range(len(df)):
                if reliable_pair.iloc[j]:
                    segment_length = np.sqrt(np.sum((p2[j] - p1[j])**2))
                    actual_lengths[j] += segment_length
                    segment_counts[j] += 1
        
        # Calculate direct distance from top to bottom
        direct_distances = np.full(len(df), np.nan)
        
        # Create a mask where both endpoints are reliable
        reliable_endpoints = reliable_keypoints[1] & reliable_keypoints[num_keypoints]
        
        if has_z:
            first = df[[f'x1', f'y1', f'z1']].values
            last = df[[f'x{num_keypoints}', f'y{num_keypoints}', f'z{num_keypoints}']].values
        else:
            first = df[[f'x1', f'y1']].values
            last = df[[f'x{num_keypoints}', f'y{num_keypoints}']].values
        
        for j in range(len(df)):
            if reliable_endpoints.iloc[j]:
                direct_distances[j] = np.sqrt(np.sum((last[j] - first[j])**2))
        
        # Calculate curvature ratio only where we have both values
        curvature_ratios = np.full(len(df), np.nan)
        for j in range(len(df)):
            if not np.isnan(direct_distances[j]) and direct_distances[j] > 0 and segment_counts[j] > 0:
                curvature_ratios[j] = actual_lengths[j] / direct_distances[j]
        
        enhanced_df['spine_curvature_ratio'] = curvature_ratios
    
    # 6. Calculate weighted visibility features
    
    # Average visibility per region
    if num_keypoints >= 9 and 'v1' in df.columns:
        # Define regions (adjust based on your keypoint definitions)
        cervical_end = min(7, num_keypoints // 3)
        thoracic_end = min(19, 2 * num_keypoints // 3)
        
        # Calculate average visibility in each region
        cervical_vis = []
        thoracic_vis = []
        lumbar_vis = []
        
        for j in range(len(df)):
            c_vis = [df[f'v{i}'].iloc[j] for i in range(1, cervical_end+1) if f'v{i}' in df.columns]
            t_vis = [df[f'v{i}'].iloc[j] for i in range(cervical_end+1, thoracic_end+1) if f'v{i}' in df.columns]
            l_vis = [df[f'v{i}'].iloc[j] for i in range(thoracic_end+1, num_keypoints+1) if f'v{i}' in df.columns]
            
            cervical_vis.append(np.mean(c_vis) if c_vis else np.nan)
            thoracic_vis.append(np.mean(t_vis) if t_vis else np.nan)
            lumbar_vis.append(np.mean(l_vis) if l_vis else np.nan)
        
        enhanced_df['cervical_visibility'] = cervical_vis
        enhanced_df['thoracic_visibility'] = thoracic_vis
        enhanced_df['lumbar_visibility'] = lumbar_vis
    
    # 7. Vertebral rotation features (for 3D data)
    
    if has_z and num_keypoints >= 3:
        # Estimate rotation around the vertical axis for each vertebra
        for i in range(1, num_keypoints-1):
            # Create a mask where three consecutive points are reliable
            reliable_triplet = (reliable_keypoints[i-1] & 
                               reliable_keypoints[i] & 
                               reliable_keypoints[i+1])
            
            # Extract coordinates
            p_prev = df[[f'x{i-1}', f'y{i-1}', f'z{i-1}']].values if i > 1 else None
            p_curr = df[[f'x{i}', f'y{i}', f'z{i}']].values
            p_next = df[[f'x{i+1}', f'y{i+1}', f'z{i+1}']].values
            
            if p_prev is not None:
                # Calculate vectors between consecutive points
                v1 = p_curr - p_prev
                v2 = p_next - p_curr
                
                # Project these vectors onto the transverse plane (xz-plane)
                v1_transverse = np.column_stack((v1[:, 0], np.zeros(len(df)), v1[:, 2]))
                v2_transverse = np.column_stack((v2[:, 0], np.zeros(len(df)), v2[:, 2]))
                
                # Calculate rotation angles
                rotation_angles = np.full(len(df), np.nan)
                
                for j in range(len(df)):
                    if reliable_triplet.iloc[j]:
                        # Normalize vectors
                        norm_v1 = np.linalg.norm(v1_transverse[j])
                        norm_v2 = np.linalg.norm(v2_transverse[j])
                        
                        if norm_v1 > 0 and norm_v2 > 0:
                            v1_norm = v1_transverse[j] / norm_v1
                            v2_norm = v2_transverse[j] / norm_v2
                            
                            # Calculate dot product
                            dot_product = np.dot(v1_norm, v2_norm)
                            
                            # Calculate angle
                            angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi
                            rotation_angles[j] = angle
                
                enhanced_df[f'vertebral_rotation_{i}'] = rotation_angles
    
    # 8. Calculate regional curvature metrics with reliability weighting
    
    # Calculate average angles in each region, weighted by visibility
    if num_keypoints >= 9:
        # Define regions (adjust based on your keypoint definitions)
        cervical_end = min(7, num_keypoints // 3)
        thoracic_end = min(19, 2 * num_keypoints // 3)
        
        # Initialize arrays for angles and weights
        cervical_angles = np.zeros(len(df))
        thoracic_angles = np.zeros(len(df))
        lumbar_angles = np.zeros(len(df))
        
        cervical_weights = np.zeros(len(df))
        thoracic_weights = np.zeros(len(df))
        lumbar_weights = np.zeros(len(df))
        
        # Collect angle data
        for i in range(1, num_keypoints-1):
            angle_col = f'angle_{i}_{i+1}_{i+2}'
            if angle_col in enhanced_df.columns:
                # Determine which region this angle belongs to
                if i < cervical_end:
                    region_angles = cervical_angles
                    region_weights = cervical_weights
                elif i < thoracic_end:
                    region_angles = thoracic_angles
                    region_weights = thoracic_weights
                else:
                    region_angles = lumbar_angles
                    region_weights = lumbar_weights
                
                # Calculate visibility-based weight for this angle
                if f'v{i}' in df.columns and f'v{i+1}' in df.columns and f'v{i+2}' in df.columns:
                    vis_weight = (df[f'v{i}'] + df[f'v{i+1}'] + df[f'v{i+2}']) / 3
                else:
                    vis_weight = pd.Series([1.0] * len(df))  # Default weight
                
                # Add weighted angle values
                for j in range(len(df)):
                    angle_val = enhanced_df[angle_col].iloc[j]
                    if not np.isnan(angle_val):
                        weight = vis_weight.iloc[j]
                        region_angles[j] += angle_val * weight
                        region_weights[j] += weight
        
        # Calculate weighted averages
        cervical_avg = np.full(len(df), np.nan)
        thoracic_avg = np.full(len(df), np.nan)
        lumbar_avg = np.full(len(df), np.nan)
        
        for j in range(len(df)):
            if cervical_weights[j] > 0:
                cervical_avg[j] = cervical_angles[j] / cervical_weights[j]
            if thoracic_weights[j] > 0:
                thoracic_avg[j] = thoracic_angles[j] / thoracic_weights[j]
            if lumbar_weights[j] > 0:
                lumbar_avg[j] = lumbar_angles[j] / lumbar_weights[j]
        
        enhanced_df['weighted_cervical_angle'] = cervical_avg
        enhanced_df['weighted_thoracic_angle'] = thoracic_avg
        enhanced_df['weighted_lumbar_angle'] = lumbar_avg
    
    # 9. Confidence metrics for the entire analysis
    
    # Calculate overall keypoint reliability score
    if any(f'v{i}' in df.columns for i in range(1, num_keypoints+1)):
        all_visibility = []
        for i in range(1, num_keypoints+1):
            if f'v{i}' in df.columns:
                all_visibility.append(df[f'v{i}'])
        
        if all_visibility:
            enhanced_df['overall_keypoint_reliability'] = pd.concat(all_visibility, axis=1).mean(axis=1)
    
    # Print summary of new features
    new_features = [col for col in enhanced_df.columns if col not in df.columns]
    print(f"Added {len(new_features)} new anatomical features:")
    for feature in new_features:
        print(f"  - {feature}")
    
    return enhanced_df
            



