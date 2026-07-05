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

model = keras.models.load_model('models/v3.2_emotion_model_lighter_smoothing.keras')

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=class_names))