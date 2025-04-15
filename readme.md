# Real-Time Facial Expression Recognition using Optimized FRR-CNN with VGGNet Transfer Learning

This project implements a real-time facial expression recognition (FER) system using an optimized FRR-CNN architecture enhanced with transfer learning from VGGNet. It classifies facial expressions into 7 categories and supports live webcam detection, screenshot saving per emotion, and evaluation on FER-2013 and external datasets like CK+ and JAFFE.

---

## 🔧 Features

- Real-time emotion detection via webcam
- Optimized FRR-CNN architecture + VGGNet transfer learning
- Screenshot saving per predicted emotion
- Evaluation on FER-2013 and external datasets
- Training plots for accuracy and loss
- Organized codebase with modular structure

---

## 🧠 Emotion Classes

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## 📁 Project Structure

```
fer_frr_cnn_project/
├── config.py                  # Hyperparameter configuration
├── train.py                   # Train the model
├── evaluate.py                # Evaluate on FER-2013 test data
├── inference.py               # Real-time webcam prediction
├── test_on_ck_jafee.py        # External dataset testing (CK+/JAFFE)
├── frr_cnn_model.h5           # Trained model file
├── ck_jafee_results.csv       # Output from CK+/JAFFE test
├── requirements.txt           # Required packages
├── models/
│   └── frr_cnn.py             # FRR-CNN + VGGNet model
├── utils/
│   ├── preprocess.py          # Data loading and preprocessing
│   └── metrics.py             # Evaluation utilities
├── data/
│   └── fer2013.csv            # FER-2013 dataset (CSV format)
├── screenshots/
│   └── angry/                 # Screenshots saved by predicted emotion
```

---

##  How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```

### 3. Evaluate on FER-2013 test set
```bash
python evaluate.py
```

### 4. Run real-time webcam FER
```bash
python inference.py
```
- Press `s` to save a screenshot to `screenshots/<emotion>/`
- Press `q` to quit

### 5. Test on CK+ or JAFFE Dataset
Ensure `data/test_ck_jafee/` is structured with subfolders like `angry/`, `happy/`, etc. Then:
```bash
python test_on_ck_jafee.py
```

---

##  Outputs

- `frr_cnn_model.h5` — Trained model file
- `ck_jafee_results.csv` — Prediction results on external dataset
- `screenshots/<emotion>/` — Real-time prediction screenshots

---

##  Dataset Links

- [FER-2013 (Kaggle)](https://www.kaggle.com/datasets/msambare/fer2013)
- [CK+ Dataset (manual request)](https://www.jeffcohn.net/resources/)
- [JAFFE Dataset](http://www.kasrl.org/jaffe_download.html)

