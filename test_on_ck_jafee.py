import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import config

# Define emotion mapping for folder names → FER-2013 label indices
emotion_map = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5,
    'neutral': 6
}

# Load trained model
model = load_model(config.MODEL_NAME)
emotion_labels = list(emotion_map.keys())

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, config.IMG_SIZE)
    img = img.astype('float32') / 255.0
    return np.expand_dims(np.expand_dims(img, -1), axis=0)

def load_images_from_folder(folder):
    X, y, paths = [], [], []
    for label_name in os.listdir(folder):
        class_path = os.path.join(folder, label_name)
        if not os.path.isdir(class_path) or label_name.lower() not in emotion_map:
            continue
        label = emotion_map[label_name.lower()]
        for filename in os.listdir(class_path):
            filepath = os.path.join(class_path, filename)
            img = preprocess_image(filepath)
            X.append(img)
            y.append(label)
            paths.append(filepath)
    return np.vstack(X), np.array(y), paths

# Load test data
X_test, y_true, image_paths = load_images_from_folder('data/test_ck_jafee')
y_pred = np.argmax(model.predict(X_test), axis=1)

# Save results
results_df = pd.DataFrame({
    'Image_Path': image_paths,
    'True_Label': y_true,
    'Predicted_Label': y_pred,
    'True_Emotion': [emotion_labels[i] for i in y_true],
    'Predicted_Emotion': [emotion_labels[i] for i in y_pred]
})
results_df.to_csv('ck_jafee_results.csv', index=False)

print("[INFO] CK+/JAFFE evaluation results saved to ck_jafee_results.csv")
