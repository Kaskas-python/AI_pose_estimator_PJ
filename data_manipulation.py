import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Augmentation

def augment_keypoint_data(df, noise_std=0.02, scale_range=(0.95, 1.05), shift_range=(-0.05, 0.05), augment_factor=2):
    """
    Augments the dataset by applying jittering, scaling, and shifting to keypoint coordinates (X, Y, Z).
    Visibility scores remain unchanged.

    Args:
        df (DataFrame): Original dataset containing keypoint features (X, Y, Z, visibility) and labels.
        noise_std (float): Standard deviation of Gaussian noise for jittering.
        scale_range (tuple): Min and max scaling factors.
        shift_range (tuple): Min and max shifting values.
        augment_factor (int): Number of augmented samples per original sample.

    Returns:
        DataFrame: Augmented dataset.
    """
    augmented_data = []

    for _ in range(augment_factor):  # Generate multiple augmented versions
        augmented_df = df.copy()

        # Identify keypoint columns
        keypoint_cols = [col for col in df.columns if col not in ['label', 'id']]
        visibility_cols = [col for col in keypoint_cols if 'visibility' in col]  # Extract visibility score columns
        xyz_cols = [col for col in keypoint_cols if col not in visibility_cols]  # Extract X, Y, Z columns

        # Jittering: Add small Gaussian noise (Not applied to visibility)
        augmented_df[xyz_cols] += np.random.normal(0, noise_std, size=augmented_df[xyz_cols].shape)

        # Scaling: Multiply by a small random factor (Not applied to visibility)
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        augmented_df[xyz_cols] *= scale_factor

        # Shifting: Add a small random value (Not applied to visibility)
        shift_values = np.random.uniform(shift_range[0], shift_range[1], size=augmented_df[xyz_cols].shape)
        augmented_df[xyz_cols] += shift_values

        # Keep visibility scores unchanged
        augmented_df[visibility_cols] = df[visibility_cols]

        augmented_data.append(augmented_df)

    return pd.concat([df] + augmented_data, ignore_index=True)

def calculate_head_shoulder_angle(df, row_index=None):
    """
    Calculate the head-to-shoulder angle for kyphosis assessment.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing the pose landmarks
    row_index (int, optional): Index of the row to process. If None, processes all rows.
    
    Returns:
    If row_index is provided, returns the angle for that specific row.
    If row_index is None, returns a Series of angles for all rows.
    """
    if row_index is not None:
        # Process a single row
        row = df.iloc[row_index]
        return _calculate_angle_from_row(row)
    else:
        # Process all rows
        return df.apply(_calculate_angle_from_row, axis=1)

def _calculate_angle_from_row(row):
    """Helper function to calculate angle from a single DataFrame row"""
    # Extract coordinates
    left_ear = np.array([row['LEFT_EAR_x'], row['LEFT_EAR_y'], row['LEFT_EAR_z']])
    right_ear = np.array([row['RIGHT_EAR_x'], row['RIGHT_EAR_y'], row['RIGHT_EAR_z']])
    left_shoulder = np.array([row['LEFT_SHOULDER_x'], row['LEFT_SHOULDER_y'], row['LEFT_SHOULDER_z']])
    right_shoulder = np.array([row['RIGHT_SHOULDER_x'], row['RIGHT_SHOULDER_y'], row['RIGHT_SHOULDER_z']])
    
    # Calculate midpoints
    ear_midpoint = (left_ear + right_ear) / 2
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2
    
    # Calculate vector from shoulder to ear
    shoulder_to_ear = ear_midpoint - shoulder_midpoint
    
    # Define vertical reference vector (pointing upward in the Y-axis)
    vertical_vector = np.array([0, 1, 0])
    
    # Calculate the angle between the shoulder-to-ear vector and the vertical vector
    cos_angle = np.dot(shoulder_to_ear, vertical_vector) / (np.linalg.norm(shoulder_to_ear) * np.linalg.norm(vertical_vector))
    angle_radians = math.acos(np.clip(cos_angle, -1.0, 1.0))
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees

def calculate_forward_head_position(df, row_index=None):
    """
    Calculate the forward head position angle for kyphosis assessment.
    This measures the angle between the vertical line and the line from the nose to the shoulder midpoint.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing the pose landmarks
    row_index (int, optional): Index of the row to process. If None, processes all rows.
    
    Returns:
    If row_index is provided, returns the angle for that specific row.
    If row_index is None, returns a Series of angles for all rows.
    """
    if row_index is not None:
        # Process a single row
        row = df.iloc[row_index]
        return _calculate_fhp_from_row(row)
    else:
        # Process all rows
        return df.apply(_calculate_fhp_from_row, axis=1)

def _calculate_fhp_from_row(row):
    """Helper function to calculate forward head position from a single DataFrame row"""
    # Extract coordinates
    nose = np.array([row['NOSE_x'], row['NOSE_y'], row['NOSE_z']])
    left_shoulder = np.array([row['LEFT_SHOULDER_x'], row['LEFT_SHOULDER_y'], row['LEFT_SHOULDER_z']])
    right_shoulder = np.array([row['RIGHT_SHOULDER_x'], row['RIGHT_SHOULDER_y'], row['RIGHT_SHOULDER_z']])
    
    # Calculate shoulder midpoint
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2
    
    # Calculate vector from shoulder to nose
    shoulder_to_nose = nose - shoulder_midpoint
    
    # Define vertical reference vector (pointing upward in the Y-axis)
    vertical_vector = np.array([0, 1, 0])
    
    # Calculate the angle between the shoulder-to-nose vector and the vertical vector
    cos_angle = np.dot(shoulder_to_nose, vertical_vector) / (np.linalg.norm(shoulder_to_nose) * np.linalg.norm(vertical_vector))
    angle_radians = math.acos(np.clip(cos_angle, -1.0, 1.0))
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees

def calculate_shoulder_alignment_angle(df, row_index=None):
    """
    Calculate the shoulder alignment angle (how level the shoulders are).
    This measures the angle between the horizontal line and the line connecting the shoulders.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing the pose landmarks
    row_index (int, optional): Index of the row to process. If None, processes all rows.
    
    Returns:
    If row_index is provided, returns the angle for that specific row.
    If row_index is None, returns a Series of angles for all rows.
    """
    if row_index is not None:
        # Process a single row
        row = df.iloc[row_index]
        return _calculate_shoulder_angle_from_row(row)
    else:
        # Process all rows
        return df.apply(_calculate_shoulder_angle_from_row, axis=1)

def _calculate_shoulder_angle_from_row(row):
    """Helper function to calculate shoulder alignment angle from a single DataFrame row"""
    # Extract coordinates
    left_shoulder = np.array([row['LEFT_SHOULDER_x'], row['LEFT_SHOULDER_y']])
    right_shoulder = np.array([row['RIGHT_SHOULDER_x'], row['RIGHT_SHOULDER_y']])
    
    # Calculate shoulder vector (from right to left)
    shoulder_vector = left_shoulder - right_shoulder
    
    # Define horizontal reference vector
    horizontal_vector = np.array([1, 0])
    
    # Calculate the angle between the shoulder vector and the horizontal vector
    cos_angle = np.dot(shoulder_vector, horizontal_vector) / (np.linalg.norm(shoulder_vector) * np.linalg.norm(horizontal_vector))
    angle_radians = math.acos(np.clip(cos_angle, -1.0, 1.0))
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees

def add_kyphosis_features(df):
    """
    Add all kyphosis-related features to the DataFrame.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing the pose landmarks
    
    Returns:
    pandas.DataFrame: The original DataFrame with added feature columns
    """
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Calculate and add all features
    result_df['head_shoulder_angle'] = calculate_head_shoulder_angle(df)
    result_df['forward_head_position'] = calculate_forward_head_position(df)
    result_df['shoulder_alignment_angle'] = calculate_shoulder_alignment_angle(df)
    
    return result_df

# Example usage:
# df = pd.read_csv('pose_landmarks.csv')
# df_with_features = add_kyphosis_features(df)
# print(df_with_features[['id', 'label', 'head_shoulder_angle', 'forward_head_position', 'shoulder_alignment_angle']])
def create_anatomical_features(df):
    """
    Create anatomical features relevant to kyphosis from raw landmark data
    """
    features_df = df.copy()
    
    # Add features we calculated earlier
    features_df['head_shoulder_angle'] = calculate_head_shoulder_angle(df)
    features_df['forward_head_position'] = calculate_forward_head_position(df)
    features_df['shoulder_alignment_angle'] = calculate_shoulder_alignment_angle(df)
    
    # Additional anatomical features
    
    # Calculate neck inclination
    features_df['neck_length'] = np.sqrt(
        ((df['NOSE_x'] - df['LEFT_SHOULDER_x'] + df['RIGHT_SHOULDER_x'])/2)**2 + 
        ((df['NOSE_y'] - df['LEFT_SHOULDER_y'] + df['RIGHT_SHOULDER_y'])/2)**2
    )
    
    # Calculate shoulder to hip vertical distance
    features_df['shoulder_hip_vertical_distance'] = np.abs(
        (df['LEFT_SHOULDER_y'] + df['RIGHT_SHOULDER_y'])/2 - 
        (df['LEFT_HIP_y'] + df['RIGHT_HIP_y'])/2
    )
    
    # Calculate shoulder width
    features_df['shoulder_width'] = np.sqrt(
        (df['LEFT_SHOULDER_x'] - df['RIGHT_SHOULDER_x'])**2 + 
        (df['LEFT_SHOULDER_y'] - df['RIGHT_SHOULDER_y'])**2
    )
    
    # Calculate mid-spine approximation 
    # (using ratio of shoulder-to-hip distance as a proxy for spine curvature)
    left_mid = (df['LEFT_SHOULDER_y'] + df['LEFT_HIP_y'])/2
    right_mid = (df['RIGHT_SHOULDER_y'] + df['RIGHT_HIP_y'])/2
    shoulder_mid_y = (df['LEFT_SHOULDER_y'] + df['RIGHT_SHOULDER_y'])/2
    hip_mid_y = (df['LEFT_HIP_y'] + df['RIGHT_HIP_y'])/2
    
    features_df['spine_curve_proxy'] = np.abs(
        (left_mid + right_mid)/2 - (shoulder_mid_y + hip_mid_y)/2
    )
    
    # Head position relative to shoulders (horizontal)
    features_df['head_forward_displacement'] = (
        df['NOSE_x'] - (df['LEFT_SHOULDER_x'] + df['RIGHT_SHOULDER_x'])/2
    )
    
    # Extract only the feature columns we've calculated
    feature_cols = [
        'head_shoulder_angle', 'forward_head_position', 'shoulder_alignment_angle',
        'neck_length', 'shoulder_hip_vertical_distance', 'shoulder_width',
        'spine_curve_proxy', 'head_forward_displacement'
    ]
    
    return features_df, feature_cols

def analyze_feature_importance(X, y, feature_names):
    """
    Analyze feature importance using multiple methods
    """
    # 1. Correlation analysis
    X_with_y = X.copy()
    X_with_y['label'] = y
    correlation_matrix = X_with_y.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # 2. ANOVA F-value for feature ranking
    selector = SelectKBest(f_classif, k='all')
    selector.fit(X, y)
    f_scores = pd.DataFrame({
        'Feature': feature_names,
        'F_Score': selector.scores_,
        'P_value': selector.pvalues_
    }).sort_values('F_Score', ascending=False)
    
    print("ANOVA F-value Feature Ranking:")
    print(f_scores)
    
    # 3. Mutual Information for feature ranking
    selector = SelectKBest(mutual_info_classif, k='all')
    selector.fit(X, y)
    mi_scores = pd.DataFrame({
        'Feature': feature_names,
        'MI_Score': selector.scores_
    }).sort_values('MI_Score', ascending=False)
    
    print("\nMutual Information Feature Ranking:")
    print(mi_scores)
    
    # 4. Random Forest feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nRandom Forest Feature Importance:")
    print(rf_importance)
    
    # 5. Recursive Feature Elimination
    model = LogisticRegression(max_iter=1000)
    rfe = RFE(estimator=model, n_features_to_select=1, step=1)
    rfe.fit(X, y)
    rfe_ranking = pd.DataFrame({
        'Feature': feature_names,
        'RFE_Rank': rfe.ranking_
    }).sort_values('RFE_Rank')
    
    print("\nRecursive Feature Elimination Ranking:")
    print(rfe_ranking)
    
    # Return a combined ranking dataframe
    combined_ranking = pd.DataFrame({'Feature': feature_names})
    combined_ranking = combined_ranking.merge(f_scores[['Feature', 'F_Score']], on='Feature')
    combined_ranking = combined_ranking.merge(mi_scores[['Feature', 'MI_Score']], on='Feature')
    combined_ranking = combined_ranking.merge(rf_importance[['Feature', 'Importance']], on='Feature')
    combined_ranking = combined_ranking.merge(rfe_ranking[['Feature', 'RFE_Rank']], on='Feature')
    
    # Calculate an overall score (simple mean of normalized ranks)
    for col in ['F_Score', 'MI_Score', 'Importance']:
        combined_ranking[f'{col}_norm'] = combined_ranking[col] / combined_ranking[col].max()
    
    combined_ranking['RFE_Rank_norm'] = 1 - ((combined_ranking['RFE_Rank'] - 1) / 
                                           (combined_ranking['RFE_Rank'].max() - 1))
    
    combined_ranking['Overall_Score'] = (
        combined_ranking['F_Score_norm'] + 
        combined_ranking['MI_Score_norm'] + 
        combined_ranking['Importance_norm'] + 
        combined_ranking['RFE_Rank_norm']
    ) / 4
    
    return combined_ranking.sort_values('Overall_Score', ascending=False)

def select_best_features(df, feature_cols, y_col='label', n_features=None):
    """
    Select the best features based on multiple methods and evaluate model performance
    """
    # Split data
    X = df[feature_cols]
    y = df[y_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Analyze feature importance
    importance_df = analyze_feature_importance(X_train, y_train, feature_cols)
    
    if n_features is None:
        # Find optimal number of features through cross-validation
        scores = []
        feature_counts = range(1, len(feature_cols) + 1)
        
        for n in feature_counts:
            top_features = importance_df['Feature'].iloc[:n].tolist()
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(random_state=42))
            ])
            
            score = cross_val_score(pipeline, X[top_features], y, cv=5, scoring='accuracy').mean()
            scores.append(score)
        
        plt.figure(figsize=(10, 6))
        plt.plot(feature_counts, scores, marker='o')
        plt.xlabel('Number of Features')
        plt.ylabel('Cross-Validation Accuracy')
        plt.title('Model Performance vs Number of Features')
        plt.grid(True)
        plt.show()
        
        # Select optimal number of features (could use other criteria instead of max)
        optimal_n = feature_counts[scores.index(max(scores))]
        print(f"Optimal number of features: {optimal_n}")
        best_features = importance_df['Feature'].iloc[:optimal_n].tolist()
    else:
        best_features = importance_df['Feature'].iloc[:n_features].tolist()
    
    print("\nSelected features:")
    for i, feature in enumerate(best_features, 1):
        print(f"{i}. {feature}")
    
    # Add selected features to a new dataframe
    final_df = df.copy()
    feature_cols_to_keep = ['id', y_col] + best_features
    final_df = final_df[feature_cols_to_keep]
    
    return final_df, best_features, importance_df

# Example usage
# df, feature_cols = create_anatomical_features(original_df)
# final_df, best_features, importance_ranking = select_best_features(df, feature_cols)