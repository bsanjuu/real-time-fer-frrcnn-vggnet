import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from datetime import datetime
import os
import config
from utils.preprocess import preprocess_face

# Config
st.set_page_config(page_title="😊 Real-Time Facial Expression Recognition")
st.title("😊 Real-Time Facial Expression Recognition")
st.write("Upload an image or use your webcam to detect emotions using FRR-CNN + VGGNet")

# Load model
model = load_model(config.MODEL_NAME, compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Setup directories
for emotion in emotion_labels:
    os.makedirs(f"screenshots/{emotion.lower()}", exist_ok=True)

# Real-Time Webcam Detection
st.subheader("🎥 Real-Time Webcam Emotion Detection")
run_webcam = st.checkbox("Enable Webcam")
save_frame = False

if run_webcam:
    frame_placeholder = st.empty()
    save_button = st.button("💾 Save Current Frame")

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    last_emotion = None
    last_frame = None

    while run_webcam:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Unable to access webcam")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            processed_face = preprocess_face(face_roi)
            result = model.predict(processed_face, verbose=0)[0]
            emotion_index = np.argmax(result)
            emotion = emotion_labels[emotion_index]
            confidence = int(result[emotion_index] * 100)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion} ({confidence}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            last_emotion = emotion
            last_frame = frame.copy()

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        if save_button and last_frame is not None and last_emotion is not None:
            filename = f"{last_emotion.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            save_path = os.path.join(f"screenshots/{last_emotion.lower()}", filename)
            cv2.imwrite(save_path, last_frame)
            st.success(f"✅ Saved frame as {save_path}")
            break

    cap.release()

# Upload Image
st.subheader("📸 Upload an Image")
uploaded_file = st.file_uploader("Upload a face image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)
    processed_img = preprocess_face(img_array)
    result = model.predict(processed_img, verbose=0)[0]
    emotion_index = np.argmax(result)
    emotion = emotion_labels[emotion_index]
    confidence = int(result[emotion_index] * 100)

    st.success(f"Prediction: {emotion}")
    st.write(f"Confidence: {confidence}%")

    if st.button("💾 Save Image"):
        save_dir = f"screenshots/{emotion.lower()}"
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{emotion.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, img_array)
        st.info(f"Image saved to `{save_path}`")
