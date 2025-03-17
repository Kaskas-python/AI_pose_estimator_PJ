import numpy as np

def standardize_pose_keypoints(keypoints, method='hip_centered', visibility_threshold=0.5):
    """
    Standardize MediaPipe pose keypoints to make them invariant to scale and position.
    
    Parameters:
    -----------
    keypoints : numpy.ndarray
        Array of shape (N, 3) or (N, 4) where N is the number of keypoints.
        Each row contains [x, y, visibility] or [x, y, z, visibility].
        MediaPipe typically outputs 33 keypoints.
    
    method : str, optional (default='hip_centered')
        Standardization method:
        - 'hip_centered': Centers around mid-hip and scales by torso length
        - 'shoulder_centered': Centers around mid-shoulder and scales by shoulder width
        - 'min_max': Min-max normalization to [0, 1] range
    
    visibility_threshold : float, optional (default=0.5)
        Threshold for considering a keypoint visible/valid for reference calculations
    
    Returns:
    --------
    numpy.ndarray
        Standardized keypoints with the same shape as input
    """
    # Make a copy to avoid modifying the original data
    std_keypoints = keypoints.copy()
    
    # Check dimensions
    has_z = (keypoints.shape[1] >= 4)
    
    # MediaPipe pose keypoint indices
    # Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    
    # Get visibility (last column)
    visibility = keypoints[:, -1]
    
    if method == 'hip_centered':
        # Use mid-hip as the origin
        valid_left_hip = visibility[LEFT_HIP] > visibility_threshold
        valid_right_hip = visibility[RIGHT_HIP] > visibility_threshold
        
        if valid_left_hip and valid_right_hip:
            # Calculate mid-hip point
            mid_hip_x = (keypoints[LEFT_HIP][0] + keypoints[RIGHT_HIP][0]) / 2
            mid_hip_y = (keypoints[LEFT_HIP][1] + keypoints[RIGHT_HIP][1]) / 2
            
            # Calculate torso length (mid-hip to mid-shoulder)
            valid_left_shoulder = visibility[LEFT_SHOULDER] > visibility_threshold
            valid_right_shoulder = visibility[RIGHT_SHOULDER] > visibility_threshold
            
            if valid_left_shoulder and valid_right_shoulder:
                mid_shoulder_x = (keypoints[LEFT_SHOULDER][0] + keypoints[RIGHT_SHOULDER][0]) / 2
                mid_shoulder_y = (keypoints[LEFT_SHOULDER][1] + keypoints[RIGHT_SHOULDER][1]) / 2
                
                # Torso length for scaling
                torso_length = np.sqrt((mid_shoulder_x - mid_hip_x)**2 + 
                                       (mid_shoulder_y - mid_hip_y)**2)
                
                if torso_length > 0:
                    # Translate all points to make mid-hip the origin
                    std_keypoints[:, 0] = (std_keypoints[:, 0] - mid_hip_x) / torso_length
                    std_keypoints[:, 1] = (std_keypoints[:, 1] - mid_hip_y) / torso_length
                    
                    # If Z coordinates are present
                    if has_z:
                        # For Z, we don't translate to origin but still scale
                        mid_hip_z = (keypoints[LEFT_HIP][2] + keypoints[RIGHT_HIP][2]) / 2
                        std_keypoints[:, 2] = (std_keypoints[:, 2] - mid_hip_z) / torso_length
            
    elif method == 'shoulder_centered':
        # Use mid-shoulder as the origin
        valid_left_shoulder = visibility[LEFT_SHOULDER] > visibility_threshold
        valid_right_shoulder = visibility[RIGHT_SHOULDER] > visibility_threshold
        
        if valid_left_shoulder and valid_right_shoulder:
            # Calculate mid-shoulder point
            mid_shoulder_x = (keypoints[LEFT_SHOULDER][0] + keypoints[RIGHT_SHOULDER][0]) / 2
            mid_shoulder_y = (keypoints[LEFT_SHOULDER][1] + keypoints[RIGHT_SHOULDER][1]) / 2
            
            # Calculate shoulder width for scaling
            shoulder_width = np.sqrt((keypoints[LEFT_SHOULDER][0] - keypoints[RIGHT_SHOULDER][0])**2 + 
                                    (keypoints[LEFT_SHOULDER][1] - keypoints[RIGHT_SHOULDER][1])**2)
            
            if shoulder_width > 0:
                # Translate all points to make mid-shoulder the origin
                std_keypoints[:, 0] = (std_keypoints[:, 0] - mid_shoulder_x) / shoulder_width
                std_keypoints[:, 1] = (std_keypoints[:, 1] - mid_shoulder_y) / shoulder_width
                
                # If Z coordinates are present
                if has_z:
                    mid_shoulder_z = (keypoints[LEFT_SHOULDER][2] + keypoints[RIGHT_SHOULDER][2]) / 2
                    std_keypoints[:, 2] = (std_keypoints[:, 2] - mid_shoulder_z) / shoulder_width
    
    elif method == 'min_max':
        # Simple min-max normalization to [0, 1] range
        # Only normalize points that are visible
        valid_points = visibility > visibility_threshold
        
        if np.sum(valid_points) > 1:  # Need at least 2 valid points
            # Get min and max for X and Y (only considering visible points)
            min_x = np.min(keypoints[valid_points, 0])
            max_x = np.max(keypoints[valid_points, 0])
            min_y = np.min(keypoints[valid_points, 1])
            max_y = np.max(keypoints[valid_points, 1])
            
            # Avoid division by zero
            x_range = max_x - min_x
            y_range = max_y - min_y
            
            if x_range > 0:
                std_keypoints[:, 0] = (std_keypoints[:, 0] - min_x) / x_range
            if y_range > 0:
                std_keypoints[:, 1] = (std_keypoints[:, 1] - min_y) / y_range
            
            # If Z coordinates are present
            if has_z:
                min_z = np.min(keypoints[valid_points, 2])
                max_z = np.max(keypoints[valid_points, 2])
                z_range = max_z - min_z
                
                if z_range > 0:
                    std_keypoints[:, 2] = (std_keypoints[:, 2] - min_z) / z_range
    
    return std_keypoints

def apply_pose_alignment(keypoints, reference_pose=None, visibility_threshold=0.5):
    """
    Aligns a pose to a reference pose orientation or to a standard orientation.
    
    Parameters:
    -----------
    keypoints : numpy.ndarray
        Array of shape (N, 3) or (N, 4) where N is the number of keypoints.
        Each row contains [x, y, visibility] or [x, y, z, visibility].
    
    reference_pose : numpy.ndarray, optional
        Reference pose for alignment. If None, aligns to vertical orientation.
    
    visibility_threshold : float, optional (default=0.5)
        Threshold for considering a keypoint visible/valid
        
    Returns:
    --------
    numpy.ndarray
        Aligned keypoints with the same shape as input
    """
    # Make a copy to avoid modifying the original data
    aligned_keypoints = keypoints.copy()
    
    # MediaPipe pose keypoint indices
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    
    # Get visibility (last column)
    visibility = keypoints[:, -1]
    
    # Check if shoulder and hip keypoints are visible
    valid_left_shoulder = visibility[LEFT_SHOULDER] > visibility_threshold
    valid_right_shoulder = visibility[RIGHT_SHOULDER] > visibility_threshold
    valid_left_hip = visibility[LEFT_HIP] > visibility_threshold
    valid_right_hip = visibility[RIGHT_HIP] > visibility_threshold
    
    if valid_left_shoulder and valid_right_shoulder and valid_left_hip and valid_right_hip:
        # Calculate the midpoints of shoulders and hips
        mid_shoulder = [(keypoints[LEFT_SHOULDER][0] + keypoints[RIGHT_SHOULDER][0]) / 2,
                        (keypoints[LEFT_SHOULDER][1] + keypoints[RIGHT_SHOULDER][1]) / 2]
        
        mid_hip = [(keypoints[LEFT_HIP][0] + keypoints[RIGHT_HIP][0]) / 2,
                   (keypoints[LEFT_HIP][1] + keypoints[RIGHT_HIP][1]) / 2]
        
        # Calculate the current spine vector (from mid-hip to mid-shoulder)
        spine_vector = [mid_shoulder[0] - mid_hip[0], mid_shoulder[1] - mid_hip[1]]
        spine_length = np.sqrt(spine_vector[0]**2 + spine_vector[1]**2)
        
        if spine_length > 0:
            # Normalize the spine vector
            spine_vector = [spine_vector[0] / spine_length, spine_vector[1] / spine_length]
            
            # If no reference pose is provided, align to vertical orientation (0, -1)
            target_vector = [0, -1]  # Pointing upward in image coordinates (y decreases upward)
            
            if reference_pose is not None:
                # Use spine vector from reference pose instead
                ref_visibility = reference_pose[:, -1]
                ref_valid_shoulders = (ref_visibility[LEFT_SHOULDER] > visibility_threshold and 
                                      ref_visibility[RIGHT_SHOULDER] > visibility_threshold)
                ref_valid_hips = (ref_visibility[LEFT_HIP] > visibility_threshold and 
                                 ref_visibility[RIGHT_HIP] > visibility_threshold)
                
                if ref_valid_shoulders and ref_valid_hips:
                    ref_mid_shoulder = [(reference_pose[LEFT_SHOULDER][0] + reference_pose[RIGHT_SHOULDER][0]) / 2,
                                       (reference_pose[LEFT_SHOULDER][1] + reference_pose[RIGHT_SHOULDER][1]) / 2]
                    
                    ref_mid_hip = [(reference_pose[LEFT_HIP][0] + reference_pose[RIGHT_HIP][0]) / 2,
                                  (reference_pose[LEFT_HIP][1] + reference_pose[RIGHT_HIP][1]) / 2]
                    
                    ref_spine = [ref_mid_shoulder[0] - ref_mid_hip[0], ref_mid_shoulder[1] - ref_mid_hip[1]]
                    ref_spine_length = np.sqrt(ref_spine[0]**2 + ref_spine[1]**2)
                    
                    if ref_spine_length > 0:
                        target_vector = [ref_spine[0] / ref_spine_length, ref_spine[1] / ref_spine_length]
            
            # Calculate the rotation angle between the current spine and target orientation
            # Using the cross product to determine rotation direction
            cos_angle = spine_vector[0] * target_vector[0] + spine_vector[1] * target_vector[1]
            sin_angle = spine_vector[0] * target_vector[1] - spine_vector[1] * target_vector[0]
            angle = np.arctan2(sin_angle, cos_angle)
            
            # Create rotation matrix
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            
            # Apply rotation to all keypoints around the mid-hip point
            for i in range(len(keypoints)):
                if visibility[i] > visibility_threshold:
                    # Translate point to origin (mid-hip is the center of rotation)
                    point = [keypoints[i][0] - mid_hip[0], keypoints[i][1] - mid_hip[1]]
                    
                    # Apply rotation
                    rotated_point = np.dot(rotation_matrix, point)
                    
                    # Translate back
                    aligned_keypoints[i][0] = rotated_point[0] + mid_hip[0]
                    aligned_keypoints[i][1] = rotated_point[1] + mid_hip[1]
    
    return aligned_keypoints

# Example usage
if __name__ == "__main__":
    # Create sample data (33 keypoints from MediaPipe with x, y, z, visibility)
    # This is just placeholder data
    sample_keypoints = np.random.rand(33, 4)
    sample_keypoints[:, 3] = 0.9  # Set high visibility for all keypoints
    
    # Standardize keypoints
    std_keypoints = standardize_pose_keypoints(sample_keypoints, method='hip_centered')
    
    # Align keypoints
    aligned_keypoints = apply_pose_alignment(sample_keypoints)
    
    print("Original shape:", sample_keypoints.shape)
    print("Standardized shape:", std_keypoints.shape)
    print("Aligned shape:", aligned_keypoints.shape)
