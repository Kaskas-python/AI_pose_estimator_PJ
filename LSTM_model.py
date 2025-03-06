
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset

def preprocess_data(df):
    # Extract features (keypoints) and labels
    features = df.drop(columns = ['label', 'id'])  # Keypoint x, y, z coordinates
    labels = df['label'] # 'normal', 'mild_kyphosis', 'severe_kyphosis'

    # Normalize keypoints
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Encode labels (convert to numeric values)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)  # Converts to 0, 1, 2

    # Reshape for LSTM (samples, time steps, features)
    features = features.reshape(features.shape[0], 1, features.shape[1])

    # Split into train & test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, features

def run_model(X_train, X_test, y_train, y_test, features):
    # Build LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(1, features.shape[2])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(3, activation="softmax")  # 3 classes: Normal, Mild Kyphosis, Severe Kyphosis
    ])

    # Compile model
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train model
    model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

    # Save model
    model.save("lumbar_kyphosis_model.h5")

    print("Model training complete and saved as 'lumbar_kyphosis_model.h5'.")
