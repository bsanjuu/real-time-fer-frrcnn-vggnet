from models.frr_cnn import build_frr_cnn
from utils.preprocess import load_fer2013
import config
import matplotlib.pyplot as plt


X_train, X_test, y_train, y_test = load_fer2013()


model = build_frr_cnn(input_shape=(48, 48, 1), num_classes=config.NUM_CLASSES)
model.summary()


history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=config.EPOCHS,
                    batch_size=config.BATCH_SIZE)


model.save(config.MODEL_NAME)


plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png')
plt.show()


plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')
plt.show()
