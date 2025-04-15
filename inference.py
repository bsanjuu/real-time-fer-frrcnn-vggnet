import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from datetime import datetime
import config

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model = load_model(config.MODEL_NAME)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Base screenshot folder
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

def preprocess_face(face):
    face = cv2.resize(face, config.IMG_SIZE)
    face = face.astype('float32') / 255.0
    return np.expand_dims(np.expand_dims(face, -1), axis=0)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # To hold latest prediction (for screenshot)
    current_emotion = "unknown"

    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y+h, x:x+w]
        processed_face = preprocess_face(face_roi)

        prediction = model.predict(processed_face, verbose=0)[0]
        emotion_idx = np.argmax(prediction)
        current_emotion = emotion_labels[emotion_idx]
        confidence = prediction[emotion_idx]

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Show label
        cv2.putText(frame, f"{current_emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Show confidence bar
        bar_x = x
        bar_y = y + h + 10
        bar_width = int(w * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + w, bar_y + 20), (255, 255, 255), 2)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (0, 255, 0), -1)
        cv2.putText(frame, f"{int(confidence*100)}%", (bar_x, bar_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Display the frame
    cv2.imshow("FER - Real-Time", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save frame in emotion-named subfolder
        emotion_folder = os.path.join(SCREENSHOT_DIR, current_emotion.lower())
        os.makedirs(emotion_folder, exist_ok=True)

        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(emotion_folder, filename)
        cv2.imwrite(path, frame)
        print(f"[INFO] Screenshot saved: {path}")

cap.release()
cv2.destroyAllWindows()
