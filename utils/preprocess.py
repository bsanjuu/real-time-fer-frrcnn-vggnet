import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_fer2013(path='data/fer2013.csv'):
    data = pd.read_csv(path)
    pixels = data['pixels'].tolist()
    faces = [np.fromstring(p, sep=' ').reshape(48, 48) for p in pixels]
    faces = np.array(faces).astype('float32') / 255.0
    faces = np.expand_dims(faces, -1)
    labels = to_categorical(data['emotion'].values, num_classes=7)

    return train_test_split(faces, labels, test_size=0.2, random_state=42)
