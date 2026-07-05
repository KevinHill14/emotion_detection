from tensorflow import keras
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Import data for training
val_ds = keras.utils.image_dataset_from_directory(
    "data/train",
    color_mode="grayscale",
    image_size=(48, 48),
    label_mode="categorical",
    validation_split=0.1,
    subset="validation",
    seed=1
)

# Import data for training
train_ds = keras.utils.image_dataset_from_directory(
    "data/train",
    color_mode="grayscale",
    image_size=(48, 48),
    label_mode="categorical",
    validation_split=0.1,
    subset="training",
    seed=1
)

# Import data for testing
test_ds = keras.utils.image_dataset_from_directory(
    "data/test",
    color_mode="grayscale",
    image_size=(48, 48),
    label_mode="categorical",
    shuffle=False,
)

# Define classes
class_names = val_ds.class_names

# Attach step to datasets (New layer)
normalization_layer = keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Define how data will be augmented to help reduce overfitting
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.1),
])

# Define class weights to punish or reward more
class_weight = {
    0: 1.150,   # angry
    1: 4.500,   # disgust
    2: 1.500,   # fear
    3: 0.754,   # happy
    4: 0.909,   # neutral
    5: 1.300,   # sad
    6: 1.137,   # surprise
}

# Define the model
model = keras.Sequential([
    keras.layers.Input(shape=(48, 48, 1)),
    data_augmentation,

    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),  # Pool on 2x2 grids

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),  

    keras.layers.Conv2D(256, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    #keras.layers.MaxPooling2D((2, 2)),  

    keras.layers.Flatten(),

    keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.6),
    keras.layers.Dense(7, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.001)),
])

# Check params
model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)

# Setup early stopping on the training
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Smooth learning
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=80,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight
)

# Save the best model if we wanted to use for later
model.save('models/v8_emotion_model_lowLSvalue_lowL.keras')

# Get true labels and predictions for the whole test set
y_true = []
y_pred = []

# Create data for cm
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# Print debug information
cm = confusion_matrix(y_true, y_pred)
print(cm)
print(classification_report(y_true, y_pred, target_names=class_names))