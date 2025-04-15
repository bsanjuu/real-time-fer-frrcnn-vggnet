#  Real-Time Facial Expression Recognition (FER) using FRR-CNN + VGGNet + Flask

This project implements a real-time **Facial Expression Recognition** system using an optimized **FRR-CNN** model with **VGGNet transfer learning** and wraps it in a Flask web application. It supports both image uploads and live webcam streaming for emotion detection.

---

##  Features

- ️ Upload image and get prediction via web UI
-  Real-time emotion recognition via webcam stream
-  FRR-CNN + VGGNet Transfer Learning for improved accuracy
- ️ Screenshots saved per predicted emotion (in `/screenshots/<emotion>/`)
-  Model training, evaluation, and result visualization
-  Tested on FER-2013 and external datasets like CK+, JAFFE

---

## Emotion Classes

- Angry  
- Disgust  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral  

---

## Project Structure

```
real-time-fer-frrcnn-vggnet/
├── app.py                     # Flask app entry point
├── config.py                  # Model and hyperparameter configs
├── train.py                   # Training pipeline
├── evaluate.py                # FER-2013 test evaluation
├── inference.py               # Standalone real-time video prediction
├── test_on_ck_jafee.py        # CK+ / JAFFE dataset evaluation
├── frr_cnn_model.h5           # Trained model
├── templates/
│   └── index.html             # Web UI
├── static/
│   └── style.css              # UI styling
├── uploads/                   # Uploaded image storage
├── screenshots/
│   └── <emotion>/            # Screenshots categorized by prediction
├── models/
│   └── frr_cnn.py             # Model architecture
├── utils/
│   ├── preprocess.py          # Preprocessing and loader
│   ├── metrics.py             # Evaluation metrics
│   └── inference_utils.py     # Webcam feed and screenshot saving
├── data/
│   └── fer2013.csv            # FER-2013 dataset
├── requirements.txt           # Required packages
```

---

## How to Run

### 1.  Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Train the Model

```bash
python train.py
```

---

### 3.  Evaluate on FER-2013

```bash
python evaluate.py
```

---

### 4.  Run Flask Web App (Image Upload + Webcam + Screenshot)

```bash
python app.py
```

- Visit: `http://127.0.0.1:5000`
- Upload image or use webcam stream
- Click "Save Screenshot" to capture current webcam frame
- Screenshot will be saved inside: `/screenshots/<predicted_emotion>/`

---

### 5.  Test on CK+/JAFFE

Structure your folder as:

```
data/test_ck_jafee/
├── angry/
├── happy/
├── ...
```

Then run:

```bash
python test_on_ck_jafee.py
```

---

##  Outputs

- `frr_cnn_model.h5` — trained model file
- `ck_jafee_results.csv` — predictions from external datasets
- `/uploads/` — uploaded images via UI
- `/screenshots/<emotion>/` — categorized webcam snapshots

---

##  Datasets Used

- [FER-2013 (Kaggle)](https://www.kaggle.com/datasets/msambare/fer2013)
- [CK+ Dataset](https://www.jeffcohn.net/resources/) (requires request)
- [JAFFE Dataset](http://www.kasrl.org/jaffe_download.html)

---

