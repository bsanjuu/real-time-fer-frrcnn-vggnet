import tensorflow as tf
from tensorflow.keras import layers, models, applications

def build_frr_cnn(input_shape=(48, 48, 1), num_classes=7):
    # Input layer (grayscale)
    inputs = layers.Input(shape=input_shape)

    # Convert to 3 channels to match VGG16 requirements
    x = layers.Conv2D(3, (3, 3), padding='same')(inputs)

    # Load VGG16 base model
    base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(48, 48, 3))
    base_model.trainable = False
    x = base_model(x)

    # Custom head
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model