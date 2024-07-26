import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping
from models.mobilenet_v2 import MobileNetV2
from utils.preprocessing import preprocess

# Load dataset
(ds_train, ds_val), ds_info = tfds.load('cats_vs_dogs', split=['train[:80%]', 'train[80%:]'], with_info=True, as_supervised=True)

# Apply the preprocessing function to the datasets
ds_train = ds_train.map(preprocess).shuffle(1000).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
ds_val = ds_val.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

# Create the model
model = MobileNetV2(input_shape=(224, 224, 3), k=2)
model.summary()

# Compile the model
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Set up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(ds_train, validation_data=ds_val, epochs=20, callbacks=[early_stopping])

# Save the trained model
model.save('mobilenet_v2_cats_vs_dogs.h5')
