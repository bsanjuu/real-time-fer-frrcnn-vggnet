import cv2
import numpy as np
from datetime import datetime
from utils.preprocess import preprocess_face
import config
import os

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directory to save screenshots
SCREENSHOT_BASE_DIR = "screenshots"
os.makedirs(SCREENSHOT_BASE_DIR, exist_ok=True)

# Global frame for saving
last_frame = None
last_emotion = None

def generate_frames(model, emotion_labels):
    global last_frame, last_emotion
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            processed_face = preprocess_face(face_roi)

            prediction = model.predict(processed_face, verbose=0)[0]
            emotion_index = np.argmax(prediction)
            emotion = emotion_labels[emotion_index]
            confidence = int(prediction[emotion_index] * 100)

            # Draw bounding box and prediction
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{emotion} ({confidence}%)', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Save last emotion for screenshot path
            last_emotion = emotion

        last_frame = frame.copy()

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def save_current_frame():
    global last_frame, last_emotion

    if last_frame is not None and last_emotion is not None:
        # Create emotion-specific folder
        emotion_dir = os.path.join(SCREENSHOT_BASE_DIR, last_emotion.lower())
        os.makedirs(emotion_dir, exist_ok=True)

        filename = f"{last_emotion.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        path = os.path.join(emotion_dir, filename)
        cv2.imwrite(path, last_frame)
        return filename
    else:
        return "No frame available"
