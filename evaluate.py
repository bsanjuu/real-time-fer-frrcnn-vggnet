from tensorflow.keras.models import load_model
from utils.preprocess import load_fer2013
from utils.metrics import evaluate_model
import config
import pandas as pd
model = load_model(config.MODEL_NAME)
_, X_test, _, y_test = load_fer2013()

evaluate_model(model, X_test, y_test)


# Generate predictions and confidence
preds = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(preds, axis=1)
confidences = np.max(preds, axis=1)

# Save to CSV
df = pd.DataFrame({
    'True_Label': y_true,
    'Predicted_Label': y_pred,
    'Confidence': confidences
})
df.to_csv('evaluation_results.csv', index=False)
print("[INFO] Evaluation results saved to evaluation_results.csv")
