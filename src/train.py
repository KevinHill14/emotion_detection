from tensorflow import keras

# Import data
train_ds = keras.utils.image_dataset_from_directory(
    "data/train",
    color_mode="grayscale",
    image_size=(48, 48),
)

# Attach step to train_ds (New layer)
normalization_layer = keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
