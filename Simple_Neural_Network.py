from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

def preprocess_data(df):
    # Extract features (keypoints) and labels
    features = df.drop(columns=['label', 'id']) 
    labels = df['label']  # 'normal', 'mild_kyphosis', 'severe_kyphosis'
    
    # Split before scaling to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=43, stratify=labels)
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    
    print(X_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print(y_train.shape)

    return X_train, X_test, y_train, y_test, X_train.shape[1]

def build_model(input_shape):

    adam_optimizer = Adam(
        learning_rate=0.001,      # Initial learning rate
        beta_1=0.9,               # Exponential decay rate for the 1st moment estimates
        beta_2=0.999,             # Exponential decay rate for the 2nd moment estimates
        epsilon=1e-07,            # Small constant for numerical stability
        amsgrad=False,            # Whether to apply the AMSGrad variant
        weight_decay=0.0001       # Weight decay regularization term
    )
    model = Sequential([
        Dense(108, activation='relu', input_shape=(input_shape,)),
        Dropout(0.4),
        BatchNormalization(),
        
        Dense(64, activation='relu'),
        Dropout(0.4),
        BatchNormalization(),
        
        Dense(32, activation='relu'),
        Dropout(0.4),
        
        
        Dense(16, activation='relu'),

        Dense(3, activation='softmax')  # 3 classes: Normal, Mild Kyphosis, Severe Kyphosis
    ])
    
    model.compile(loss='sparse_categorical_crossentropy',
                   optimizer= adam_optimizer,
                     metrics= ['accuracy'])
    return model

def run_model(X_train, X_test, y_train, y_test, input_shape):
    model = build_model(input_shape)
    model.summary()
    # plot_model(model, to_file="model_structure.png", show_shapes=True, show_layer_names=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),
                        callbacks=[early_stopping, reduce_lr])
    
    # Save model
    model.save("lumbar_kyphosis_model_optimized.h5")
    print("Model training complete and saved as 'lumbar_kyphosis_model_optimized.h5'.")
    
    # Plot Training Accuracy and Validation Loss
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')
    
    plt.show()