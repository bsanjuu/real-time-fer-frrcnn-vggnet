import tensorflow as tf
from tensorflow.keras import layers, models, applications

def build_frr_cnn(input_shape=(48, 48, 1), num_classes=7):
    base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(48, 48, 3))
    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(3, (3, 3), padding='same'),
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
