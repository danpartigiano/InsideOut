import cv2
import numpy as np
import tensorflow as tf

###########################
# Configurations
###########################
MODEL_PATH = 'emotion_recognition_model_final.h5'  
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
TARGET_SIZE = (48, 48)   # Should match the model's expected input image size
COLOR_MODE = 'grayscale' # Because model was trained on grayscale images
EMOTIONS = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

###########################
# Load the model
###########################
model = tf.keras.models.load_model(MODEL_PATH)

###########################
# Utilities
###########################
def load_face_cascade(cascade_path):
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise IOError("Could not load Haar cascade. Ensure the file is in the correct path.")
    return face_cascade

def preprocess_face(roi, target_size, color_mode='grayscale'):
    """
    Preprocess the face ROI for model prediction:
    - Resize to target size
    - Normalize to [0,1]
    - Expand dimensions as needed by model
    """
    roi_resized = cv2.resize(roi, target_size)
    roi_resized = roi_resized.astype("float") / 255.0
    
    # If grayscale, shape will be (48,48)
    # Expand dims to (48,48,1) and then (1,48,48,1)
    if color_mode == 'grayscale':
        roi_resized = np.expand_dims(roi_resized, axis=-1)
    roi_resized = np.expand_dims(roi_resized, axis=0)
    
    return roi_resized

def predict_emotion(face_img):
    """Predict the emotion from the preprocessed face image."""
    preds = model.predict(face_img, verbose=0)
    emotion_idx = np.argmax(preds[0])
    emotion_label = EMOTIONS[emotion_idx]
    confidence = preds[0][emotion_idx]
    return emotion_label, confidence

###########################
# Main Program
###########################
def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open the webcam")

    # Load Haar Cascade for face detection
    face_cascade = load_face_cascade(FACE_CASCADE_PATH)

    print("Press 'q' to exit the application.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if COLOR_MODE == 'grayscale':
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_to_detect = gray_frame
        else:
            frame_to_detect = frame

        # Detect faces
        faces = face_cascade.detectMultiScale(frame_to_detect, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))

        for (x, y, w, h) in faces:
            # Extract the face ROI
            roi = frame_to_detect[y:y+h, x:x+w]

            # Preprocess the face
            face_img = preprocess_face(roi, TARGET_SIZE, COLOR_MODE)

            # Predict emotion
            emotion_label, confidence = predict_emotion(face_img)

            # Draw bounding box and label on the original frame (color)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{emotion_label} ({confidence:.2f})"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (36,255,12), 2)

        cv2.imshow('Emotion Recognition', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
