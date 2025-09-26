import cv2
import mediapipe as mp
import csv
import os

# Create a folder to store the collected data if it doesn't exist
data_folder = 'data'
os.makedirs(data_folder, exist_ok=True)

# Define the gesture you want to collect data for
gesture_name = input("Enter the name of the gesture you want to collect data for (e.g., peace_sign, fist): ")
csv_file_path = os.path.join(data_folder, f'{gesture_name}.csv')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --- NEW: Batch Collection Variables for Efficiency ---
IS_CAPTURING = False
TARGET_BATCH_SIZE = 50  # Collect 50 frames automatically when capture is triggered
current_batch_count = 0
# ----------------------------------------------------


# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- UPDATED INSTRUCTIONS ---
print("\n--- Data Collection Instructions ---")
print(f"Current Gesture: '{gesture_name}'")
print(f"Goal: Collect {TARGET_BATCH_SIZE} frames per batch.")
print("Press 'c' to start collecting a batch of data (hold the pose steady).")
print("Press 'q' to quit.")
print("----------------------------------\n")


# Open the CSV file to write data
with open(csv_file_path, 'a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write the header row for the CSV file
    # There are 21 hand landmarks, each with x, y, z coordinates
    if os.path.getsize(csv_file_path) == 0:
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        csv_writer.writerow(header)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Flip the frame horizontally for a more intuitive view
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Hands
        results = hands.process(image_rgb)
        
        # Capture current key press
        key = cv2.waitKey(1) & 0xFF

        # --- NEW: Check for Capture Start Key ---
        if key == ord('c') and not IS_CAPTURING:
            IS_CAPTURING = True
            current_batch_count = 0
            print(f"--- STARTED Batch Collection ({TARGET_BATCH_SIZE} frames) ---")
        # ----------------------------------------

        # --- Visualization Feedback ---
        feedback_text = ""
        color = (0, 255, 0) # Green (Ready)
        if IS_CAPTURING:
            remaining = TARGET_BATCH_SIZE - current_batch_count
            feedback_text = f"CAPTURING... {remaining} frames left!"
            color = (0, 0, 255) # Red (Recording)
        else:
            feedback_text = "Ready (Press 'c' to Capture)"

        # Display feedback text on the frame
        cv2.putText(frame, feedback_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        # ------------------------------


        # Draw hand landmarks on the frame AND handle saving
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # --- NEW: Automatic Collection Logic ---
                if IS_CAPTURING:
                    if current_batch_count < TARGET_BATCH_SIZE:
                        landmarks_list = [gesture_name]
                        for landmark in hand_landmarks.landmark:
                            # Append the normalized x, y, z coordinates
                            landmarks_list.extend([landmark.x, landmark.y, landmark.z])
                        
                        csv_writer.writerow(landmarks_list)
                        current_batch_count += 1
                        
                        if current_batch_count % 10 == 0:
                             print(f"Collected {current_batch_count}/{TARGET_BATCH_SIZE} frames...")

                    else:
                        # Batch complete
                        IS_CAPTURING = False
                        print(f"--- BATCH COMPLETE: {TARGET_BATCH_SIZE} frames collected ---")
                        # Note: Calculating total samples here is tricky due to file I/O, rely on final report.
                # ---------------------------------------

        # Display the frame
        cv2.imshow('Hand Gesture Data Collector', frame)

        # Press 'q' to quit the program
        if key == ord('q'):
            break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

# Final calculation of total samples (approximate, relying on simple division for efficiency)
try:
    # Read row count from the CSV file
    with open(csv_file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        row_count = sum(1 for row in reader) - 1 # Subtract the header row
    final_total = max(0, row_count)
except FileNotFoundError:
    final_total = 0

print(f"\nData collection complete. Final count for '{gesture_name}': {final_total} samples.")
print(f"Data saved to '{csv_file_path}'.")

