from tensorflow import keras
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

test_ds = keras.utils.image_dataset_from_directory(
    "data/test",
    color_mode="grayscale",
    image_size=(48, 48),
    label_mode="categorical",
    shuffle=False,
)
class_names = test_ds.class_names

normalization_layer = keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

model_paths = [
    'models/v2_emotion_model_baseline.keras',
    'models/v3_emotion_model_lighter_smoothing.keras',
    'models/v6_emotion_model_middleLvalue.keras',
    'models/v4_emotion_model_smallerLvalue.keras',
    'models/v5_emotion_model_smallerDvalue.keras',
]
models = [keras.models.load_model(path) for path in model_paths]

y_true = []
y_pred = []

for images, labels in test_ds:
    batch_preds = [model.predict(images, verbose=0) for model in models]
    avg_preds = np.mean(batch_preds, axis=0)

    y_true.extend(np.argmax(labels, axis=1))
    y_pred.extend(np.argmax(avg_preds, axis=1))

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=class_names))