import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import tensorflow as tf
import os
from datetime import datetime

class PostureAnalyzer:
    def __init__(self, model_path, confidence_threshold=0.5):
        # Load the pre-trained model
        self.model = tf.keras.models.load_model(model_path)
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Get the feature names used during training
        self.feature_names = self._get_feature_names()
        
        # Initialize results directory
        self.results_dir = "posture_analysis_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Class labels
        self.class_labels = ["normal", "mild_kyphosis", "severe_kyphosis"]
        
        # Posture recommendations database
        self.recommendations = {
            "normal": [
                "Your posture looks good! Continue maintaining this alignment.",
                "Remember to take regular breaks from sitting.",
                "Practice daily stretching to maintain your good posture.",
                "Consider strength training to further support your posture."
            ],
            "mild_kyphosis": [
                "Avoid hunching forward when sitting or standing.",
                "Position your computer screen at eye level to reduce neck strain.",
                "Perform chin tucks throughout the day to strengthen neck muscles.",
                "Try doorway chest stretches to open up the chest and shoulders.",
                "Consider thoracic spine extension exercises like foam rolling.",
                "Strengthen your core and back muscles with planks and rows.",
                "Take frequent posture breaks - set a timer to check your alignment.",
                "Consider consulting with a physical therapist for personalized exercises."
            ],
             "severe_kyphosis": [
                "Consult a doctor or physical therapist for a tailored rehabilitation plan",
                "Perform gentle back extension exercises (prone press-ups, cat-cow stretch)",
                "Strengthen postural muscles (lower traps, rhomboids, spinal erectors)",
                "Engage in low-impact activities like swimming or walking.",
                "Use lumbar support cushions while sitting.",
                "If necessary, wear a prescribed back brace to slow progression.",
                "Avoid high-impact activities that strain the spine",
                "Avoid prolonged sitting without lumbar support."
            ]
        }
        
        # Initialize metrics for tracking posture over time
        self.posture_history = []
        self.session_start_time = time.time()
    
    def _get_feature_names(self):
       
        # Define the landmarks used for posture analysis
        # This should match exactly what was used during training
        landmarks = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]
        
        # Create feature names for each landmark's x, y, z coordinates
        features = []
        for landmark in landmarks:
            features.extend([f"{landmark}_x", f"{landmark}_y", f"{landmark}_z", f"{landmark}_visibility"])
        
        return features
    
    def extract_landmarks(self, frame):
        """
        Extract pose landmarks from a frame using MediaPipe.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            tuple: (landmarks_dict, success)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(frame_rgb)
        
        # Draw the pose landmarks on the image
        annotated_frame = frame.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # If no pose detected, return None
        if not results.pose_landmarks:
            return annotated_frame, None, False
        
        # Extract landmarks to a dictionary
        landmarks_dict = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmark_name = self.mp_pose.PoseLandmark(idx).name.lower()
            landmarks_dict[f"{landmark_name}_x"] = landmark.x
            landmarks_dict[f"{landmark_name}_y"] = landmark.y
            landmarks_dict[f"{landmark_name}_z"] = landmark.z
            landmarks_dict[f"{landmark_name}_visibility"] = landmark.visibility
        
        return annotated_frame, landmarks_dict, True

    def preprocess_landmarks(self, landmarks_dict):
        """
        Preprocess landmarks to match the model's expected input format.
        
        Args:
            landmarks_dict (dict): Dictionary of landmark coordinates
            
        Returns:
            numpy.ndarray: Processed features ready for model input
        """
        # Create a dataframe with the same structure as training data
        features = pd.DataFrame([landmarks_dict])
        
        # Ensure we have all the required features in the correct order
        for feature in self.feature_names:
            if feature not in features.columns:
                features[feature] = 0.0
        
        # Select only the features used during training, in the correct order
        features = features[self.feature_names]
        
        # Convert to numpy array and reshape for LSTM if needed
        features_array = features.values.astype('float32')
        
        # If your model expects a sequence, reshape accordingly
        # For example, if your model expects [batch_size, time_steps, features]:
        if len(self.model.input_shape) > 2:
            time_steps = self.model.input_shape[1]
            features_array = features_array.reshape(1, time_steps, -1)
        else:
            # For non-sequence models
            features_array = features_array.reshape(1, -1)
        
        return features_array
    
    
    def predict_posture(self, landmarks_dict):
        """
        Predict posture using the pre-trained model.
        
        Args:
            landmarks_dict (dict): Dictionary of landmark coordinates
            
        Returns:
            tuple: (prediction_label, confidence)
        """
        # Preprocess landmarks
        features = self.preprocess_landmarks(landmarks_dict)
        
        # Make prediction
        prediction = self.model.predict(features)
        prediction = prediction[0]
        
        # Get the predicted class and confidence
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        
        
        # Add to posture history for tracking
        self.posture_history.append({
            'timestamp': time.time(),
            'posture_label': self.class_labels[predicted_class],
            'confidence': confidence,
        })
        
        return self.class_labels[predicted_class], confidence
    
    def get_recommendations(self, predicted_class):
        """
        Get personalized recommendations based on the predicted posture.
        
        Args:
            predicted_class (str): Predicted posture label
            
        Returns:
            list: List of recommended exercises
        """
        return self.recommendations[predicted_class]
    
    def analyze_frame(self, frame):
        """
        Analyze a frame for posture estimation.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            tuple: (annotated_frame, posture_label, confidence, recommendations, success)
        """
        # Extract landmarks
        annotated_frame, landmarks_dict, success = self.extract_landmarks(frame)
        
        if not success:
            # Add text to the frame if no pose detected
            cv2.putText(
                annotated_frame, "No pose detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
            return annotated_frame, "No pose detected", 0.0, [], False
        
        # Predict posture
        posture_label, confidence = self.predict_posture(landmarks_dict)
        
        # Get personalized recommendations
        recommendations = self.get_recommendations(posture_label)
        
        # Add text to the frame for posture label and confidence
        text = f"{posture_label}: {confidence:.2f}"
        cv2.putText(
            annotated_frame, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, 
            (0, 255, 0) if posture_label == "normal" else (0, 0, 255), 2
        )
        
        # Add recommendations to the frame
        annotated_frame = self.display_recommendations(annotated_frame, recommendations)
        
        return annotated_frame, posture_label, confidence, recommendations, success
        
    def save_analysis(self, posture_label, confidence, recommendations, frame, annotated_frame):
        """
        Save the analysis results and frame.
        
        Args:
            frame (numpy.ndarray): Annotated frame
            posture_label (str): Predicted posture label
            confidence (float): Prediction confidence
            metrics (dict): Dictionary of posture metrics
            recommendations (list): List of recommendations
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed analysis as text
        report_path = os.path.join(self.results_dir, f"posture_report_{timestamp}.txt")
        with open(report_path, 'w') as f:
            f.write("==== POSTURE ANALYSIS REPORT ====\n")
            f.write(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Posture Classification: {posture_label} (Confidence: {confidence:.2f})\n\n")
            
            f.write("\n==== RECOMMENDATIONS ====\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")

        report_path_image= os.path.join(self.results_dir, f"posture_report_{timestamp}_image.jpeg",)
        cv2.imwrite(report_path_image, annotated_frame)
        
        # Log to CSV
        log_path = os.path.join(self.results_dir, "posture_analysis_log.csv")
        log_exists = os.path.exists(log_path)
        
        with open(log_path, 'a') as f:
            if not log_exists:
                f.write(f"timestamp,posture_label,confidence \n")

            f.write(f"{timestamp},{posture_label},{confidence:.4f}\n")
        
        print(f"Analysis saved: {report_path}")
        return report_path
    
    def generate_session_summary(self):
        """
        Generate a summary of the posture analysis session.
        
        Returns:
            str: Session summary text
        """
        if not self.posture_history:
            return "No posture data recorded in this session."
        
        # Calculate session duration
        session_duration = time.time() - self.session_start_time
        hours, remainder = divmod(session_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Count posture types
        posture_counts = {}
        for entry in self.posture_history:
            label = entry['posture_label']
            posture_counts[label] = posture_counts.get(label, 0) + 1
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in self.posture_history[0]['metrics'].keys():
            values = [entry['metrics'][metric] for entry in self.posture_history]
            avg_metrics[metric] = sum(values) / len(values)
        
        # Generate summary text
        summary = "==== POSTURE ANALYSIS SESSION SUMMARY ====\n\n"
        summary += f"Session duration: {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
        summary += f"Total frames analyzed: {len(self.posture_history)}\n\n"
        
        summary += "Posture Classification Distribution:\n"
        for label, count in posture_counts.items():
            percentage = (count / len(self.posture_history)) * 100
            summary += f"- {label}: {count} frames ({percentage:.1f}%)\n"
        
        # Generate key recommendations
        if self.posture_history:
            # Get most frequent posture
            most_common_posture = max(posture_counts.items(), key=lambda x: x[1])[0]
            
            summary += "\nKey Recommendations:\n"
            recommendations = self.recommendations
            for i, rec in enumerate(recommendations, 1):
                summary += f"{i}. {rec}\n"
        
        return summary
    
    def display_recommendations(self, frame, recommendations, start_y=150):
        """
        Display recommendations on the frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            recommendations (list): List of recommendations
            start_y (int): Starting y-position for text
            
        Returns:
            numpy.ndarray: Frame with recommendations
        """
        # Create a copy of the frame
        display_frame = frame.copy()
        
        # Add a semi-transparent overlay for better text readability
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, start_y-40), (display_frame.shape[1], start_y + 40*len(recommendations)), 
                    (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        # Add title
        cv2.putText(
            display_frame, "Recommendations:", (10, start_y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        # Add recommendations
        for i, rec in enumerate(recommendations[:3]):  # Show top 3 recommendations
            # Truncate recommendation if too long
            if len(rec) > 60:
                rec = rec[:57] + "..."
                
            cv2.putText(
                display_frame, f"{i+1}. {rec}", (20, start_y + i*40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )
        
        return display_frame
    
    def run_camera(self, camera_id=0, save_interval=10):
        """
        Run the analyzer on the webcam feed.
        
        Args:
            camera_id (int): Camera ID (default: 0)
            save_interval (int): Interval in seconds to save analysis (0 to disable)
        """
        # Initialize webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        # Set resolution (adjust as needed)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Variables for FPS calculation and saving interval
        prev_time = time.time()
        last_save_time = time.time()
        frame_count = 0
        
        # UI state variables
        detailed_report_path = None
        report_display_time = 0
        
        print("Starting posture analysis. Controls:")
        print("  'q' - Quit")
        print("  's' - Save current analysis")
        print("  'd' - Generate detailed report")
        
        while cap.isOpened():
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Analyze frame
            annotated_frame, posture_label, confidence, recommendations, success = self.analyze_frame(frame)
            
            # Calculate FPS
            current_time = time.time()
            frame_count += 1
            
            if current_time - prev_time >= 1.0:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time
                
                # Add FPS text to frame
                cv2.putText(
                    annotated_frame, f"FPS: {fps:.1f}", (10, annotated_frame.shape[0] - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )
            
            # Display status
            status_text = "Status: Analyzing posture"
            if not success:
                status_text = "Status: No pose detected, please stand in frame"
            elif detailed_report_path and current_time - report_display_time < 5:  # Show for 5 seconds
                status_text = f"Report saved: {detailed_report_path}"
                
            cv2.putText(
                annotated_frame, status_text, (10, annotated_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            
            # Add controls reminder at the bottom
            controls_text = "Controls: [q]Quit [s]Save [d]Detailed Report"
            cv2.putText(
                annotated_frame, controls_text, (10, annotated_frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
            )
            
            # Display frame
            cv2.imshow('Posture Analysis', annotated_frame)
            
            # Auto-save at intervals
            if save_interval > 0 and success and (current_time - last_save_time >= save_interval):
                self.save_analysis(posture_label, confidence, recommendations, frame, annotated_frame)
                last_save_time = current_time
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and success:
                detailed_report_path = self.save_analysis(posture_label, confidence, recommendations)
                report_display_time = current_time
            elif key == ord('d') and len(self.posture_history) > 0:
                # Generate and save session summary
                summary = self.generate_session_summary()
                summary_path = os.path.join(self.results_dir, f"session_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                with open(summary_path, 'w') as f:
                    f.write(summary)
                detailed_report_path = summary_path
                report_display_time = current_time
                print(f"Session summary saved: {summary_path}")
                print("\n" + summary)
                
def main():
    """
    Main function to run the posture analyzer.
    """
    # Path to your pre-trained model
    model_path = "lumbar_kyphosis_model_optimized.h5"
    
    # Create analyzer instance
    analyzer = PostureAnalyzer(model_path)
    
    # Run the camera analysis
    # Set save_interval=0 to disable auto-saving
    analyzer.run_camera(camera_id=0, save_interval=30)


if __name__ == "__main__":
    main()