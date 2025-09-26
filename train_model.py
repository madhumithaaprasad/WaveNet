import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- Configuration ---
DATA_FOLDER = 'data'
GESTURES = ['peace_sign', 'fist', 'open_hand']
NUM_CLASSES = len(GESTURES)
# The data consists of 21 landmarks, each having (x, y, z) coordinates.
NUM_FEATURES = 21 * 3 
# ---------------------

def load_and_preprocess_data():
    """
    Loads all gesture data from the CSV files, combines them,
    and prepares the data for training.
    """
    print("--- 1. Loading and Combining Data ---")
    all_data = []

    # 1. Load data for each defined gesture
    for gesture_name in GESTURES:
        file_path = os.path.join(DATA_FOLDER, f'{gesture_name}.csv')
        if not os.path.exists(file_path):
            print(f"Warning: Data file not found for {gesture_name} at {file_path}")
            continue
        
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        all_data.append(df)
        print(f"Loaded {len(df)} samples for '{gesture_name}'.")

    if not all_data:
        print("Error: No data files found. Please ensure your CSV files are in the 'data' folder.")
        return None, None, None, None

    # Combine all DataFrames into one master DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total samples loaded: {len(combined_df)}")

    # 2. Separate features (X) and labels (y)
    # X contains all columns except the first one ('label')
    X = combined_df.iloc[:, 1:].values.astype('float32') 
    # y contains only the first column ('label')
    y = combined_df['label'].values

    # 3. Encode Labels (Mapping strings to numbers)
    # Create a mapping dictionary: {'peace_sign': 0, 'fist': 1, 'open_hand': 2}
    label_map = {name: i for i, name in enumerate(GESTURES)}
    y_encoded = np.array([label_map[label] for label in y])

    # 4. Convert to One-Hot Encoding (e.g., 2 -> [0, 0, 1])
    # This format is required for training a neural network for classification
    y_categorical = to_categorical(y_encoded, num_classes=NUM_CLASSES)
    
    print(f"Features (X) shape: {X.shape}")
    print(f"Labels (y) shape: {y_categorical.shape}")

    # 5. Split Data into Training and Testing Sets
    # We use 80% for training and 20% for testing (to evaluate performance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.20, random_state=42, stratify=y_encoded
    )

    print(f"Data split: Training ({len(X_train)} samples), Testing ({len(X_test)} samples)")
    return X_train, X_test, y_train, y_test

def build_model():
    """
    Defines the architecture of the neural network model.
    """
    print("\n--- 2. Building Neural Network Model ---")
    model = Sequential([
        # Input layer: 63 features (21 landmarks * 3 coordinates)
        Dense(128, activation='relu', input_shape=(NUM_FEATURES,)),
        Dropout(0.2), # Dropout helps prevent overfitting
        
        # Hidden layer 1
        Dense(64, activation='relu'),
        Dropout(0.2),

        # Output layer: 3 classes (peace_sign, fist, open_hand)
        # 'softmax' ensures the output sums to 1 (like probabilities)
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile the model with settings appropriate for classification
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy', # Standard loss function for multi-class classification
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def train_and_save_model(model, X_train, y_train, X_test, y_test):
    """
    Trains the model and saves the trained weights.
    """
    print("\n--- 3. Training Model (This may take a minute) ---")
    
    # Fit the model to the training data
    history = model.fit(
        X_train, y_train,
        epochs=30, # Number of training cycles
        batch_size=32, # Number of samples processed before the model is updated
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # 4. Evaluate and Save Model
    print("\n--- 4. Evaluating Model Performance ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Save the model architecture and weights for later use
    model_save_path = 'gesture_model.h5'
    model.save(model_save_path)
    print(f"Model saved successfully as '{model_save_path}'")
    
    print("\n--- Training Complete ---")


if __name__ == '__main__':
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    if X_train is not None:
        # Define the neural network structure
        model = build_model()
        
        # Train and save the model
        train_and_save_model(model, X_train, y_train, X_test, y_test)
