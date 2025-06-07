import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import time
import subprocess # To run the training script
import sys # To pass arguments or determine script path

# --- Configuration ---
# Path to the trained Keras model (output from emotiondriving.py)
DEFAULT_MODEL_DIR = r"C:\Users\HP\Desktop\desk"
DEFAULT_MODEL_NAME = "CNN_Model_emotion_trained.h5"
MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, DEFAULT_MODEL_NAME)

# Path to the training script
TRAINING_SCRIPT_PATH = r"emotiondriving.py" # Assuming it's in the same directory or on PATH
# If emotiondriving.py is in a specific location, provide the full path:
# TRAINING_SCRIPT_PATH = r"C:\path\to\your\emotiondriving.py"


# Path to the Haar cascade file for face detection
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"
if not os.path.exists(HAAR_CASCADE_PATH):
    potential_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    if os.path.exists(potential_path):
        HAAR_CASCADE_PATH = potential_path
    else:
        print(f"Error: Haar cascade file not found at the default location or in cv2.data.haarcascades.")
        print("Please download 'haarcascade_frontalface_default.xml' and place it here or update HAAR_CASCADE_PATH.")
        # exit() # We might not exit immediately if user chooses to train

# Emotion labels (CRITICAL: Order MUST match the training data's class_indices)
# Verify this from the output of your 'emotiondriving.py' script.
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Image size for the model
IMG_SIZE = 48

def run_training_script():
    """Runs the emotiondriving.py training script."""
    print("\n--- Starting Model Training ---")
    print(f"Executing: python {TRAINING_SCRIPT_PATH}")
    try:
        # Ensure the working directory for the training script is where it expects data/outputs
        # If TRAINING_SCRIPT_PATH is a full path, we can try to deduce its directory
        training_script_dir = os.path.dirname(TRAINING_SCRIPT_PATH) if os.path.dirname(TRAINING_SCRIPT_PATH) else "."

        process = subprocess.Popen([sys.executable, TRAINING_SCRIPT_PATH], cwd=training_script_dir, shell=False)
        process.wait() # Wait for the training to complete
        if process.returncode == 0:
            print("--- Model Training Script Completed Successfully ---")
            return True
        else:
            print(f"--- Model Training Script Failed (return code: {process.returncode}) ---")
            return False
    except FileNotFoundError:
        print(f"Error: Training script '{TRAINING_SCRIPT_PATH}' not found.")
        print("Please ensure the path is correct and the script exists.")
        return False
    except Exception as e:
        print(f"An error occurred while trying to run the training script: {e}")
        return False

def detect_emotion_realtime(model_to_use, face_detector):
    """Handles real-time emotion detection using the provided model and face detector."""
    print("\nInitializing webcam for real-time detection...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting real-time emotion detection. Press 'q' to quit.")
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        curr_time = time.time()
        fps = 0
        if prev_time > 0:
            time_diff = curr_time - prev_time
            if time_diff > 0:
                fps = 1 / time_diff
        prev_time = curr_time

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y+h, x:x+w]
            if roi_gray.size == 0:
                continue

            try:
                roi_gray_resized = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            except cv2.error:
                continue

            roi_normalized = roi_gray_resized.astype("float") / 255.0
            roi_input = img_to_array(roi_normalized)
            roi_input = np.expand_dims(roi_input, axis=0)

            try:
                prediction = model_to_use.predict(roi_input, verbose=0)
            except Exception as e:
                print(f"Error during prediction: {e}")
                continue

            if prediction is not None and len(prediction) > 0:
                max_index = np.argmax(prediction[0])
                emotion_label = EMOTION_LABELS[max_index] if 0 <= max_index < len(EMOTION_LABELS) else "Error"
                confidence = prediction[0][max_index]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label_text = f"{emotion_label}: {confidence*100:.1f}%"
                cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Real-time Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed.")


if __name__ == "__main__":
    print("Emotion Recognition System")
    print("--------------------------")

    action = ""
    while action not in ['1', '2']:
        print("\nChoose an action:")
        print("1. Train a new model (runs emotiondriving.py)")
        print("2. Use an existing model for real-time detection")
        action = input("Enter your choice (1 or 2): ").strip()

        if action not in ['1', '2']:
            print("Invalid choice. Please enter 1 or 2.")

    # Load Haar Cascade (needed for detection, good to check early)
    face_cascade_detector = None
    try:
        face_cascade_detector = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if face_cascade_detector.empty():
            print(f"Error: Could not load Haar cascade from {HAAR_CASCADE_PATH}.")
            if action == '2': # Critical if we are going straight to detection
                print("Cannot proceed with detection without Haar cascade. Exiting.")
                exit()
            else: # If training, it's a warning, but training script might not need it.
                 print("Warning: Haar cascade not found. Detection might fail if training is skipped.")
        else:
            print(f"Face cascade loaded successfully from {HAAR_CASCADE_PATH}")
    except Exception as e:
        print(f"Error loading face cascade: {e}")
        if action == '2':
            exit()


    if action == '1':
        training_successful = run_training_script()
        if not training_successful:
            print("Training failed. Cannot proceed with detection using a newly trained model.")
            use_existing_anyway = input(f"Do you want to try using a pre-existing model at '{MODEL_PATH}' (if any)? (yes/no): ").strip().lower()
            if use_existing_anyway != 'yes':
                exit()
        # After training, we assume the model is saved at MODEL_PATH by emotiondriving.py
        print(f"\nAttempting to load the model (newly trained or existing) from: {MODEL_PATH}")
        try:
            emotion_model_loaded = load_model(MODEL_PATH)
            print("Model loaded successfully.")
            if face_cascade_detector and not face_cascade_detector.empty():
                detect_emotion_realtime(emotion_model_loaded, face_cascade_detector)
            else:
                print("Face detector (Haar cascade) not loaded. Cannot start real-time detection.")
        except Exception as e:
            print(f"Error loading the model from {MODEL_PATH}: {e}")
            print("Please ensure the model was trained and saved correctly, or that a valid pre-trained model exists at the path.")

    elif action == '2':
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file not found at {MODEL_PATH}")
            print("Please train a model first (choose option 1) or ensure the path is correct.")
            exit()
        if face_cascade_detector is None or face_cascade_detector.empty():
            print("Face detector (Haar cascade) not loaded. Cannot start real-time detection.")
            exit()

        print(f"\nLoading existing model from: {MODEL_PATH}")
        try:
            emotion_model_loaded = load_model(MODEL_PATH)
            print("Existing model loaded successfully.")
            detect_emotion_realtime(emotion_model_loaded, face_cascade_detector)
        except Exception as e:
            print(f"Error loading existing model: {e}")
            print("Please ensure a valid pre-trained model exists at the path.")

    print("\nApplication finished.")