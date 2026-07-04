from tensorflow import keras

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

# Attach step to datasets (New layer)
normalization_layer = keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Define how data will be augmented to help reduce overfitting
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.1),
])

# Define the model
model = keras.Sequential([
    keras.layers.Input(shape=(48, 48, 1)),

    data_augmentation,

    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),  # Pool on 2x2 grids

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Flatten(),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(7, activation='softmax'),
])

# Check params
model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Setup early stopping on the training
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[early_stop]
)

model.save('models/emotion_model_v2_50epochs.h5')