from flask import Flask, render_template, request, url_for, Response, jsonify, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from datetime import datetime
from utils.preprocess import preprocess_face
from utils.inference_utils import generate_frames, save_current_frame
import config

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Emotion labels matching your model's output
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load pre-trained model
model = load_model(config.MODEL_NAME)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    filename = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Save uploaded file
            filename = datetime.now().strftime('%Y%m%d_%H%M%S_') + file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Read and preprocess
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                processed_img = preprocess_face(img)

                # Predict
                result = model.predict(processed_img, verbose=0)[0]
                emotion_index = np.argmax(result)
                prediction = emotion_labels[emotion_index]
                confidence = int(result[emotion_index] * 100)

    return render_template('index.html',
                           prediction=prediction,
                           confidence=confidence,
                           filename=filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves uploaded images to browser"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/video_feed')
def video_feed():
    """Live webcam video stream"""
    return Response(generate_frames(model, emotion_labels),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/save_screenshot', methods=['POST'])
def save_screenshot():
    """Save screenshot from live webcam feed"""
    filename = save_current_frame()
    return jsonify({"message": f"Screenshot saved as {filename}"})


if __name__ == '__main__':
    app.run(debug=False)
