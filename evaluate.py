import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_datasets as tfds
from utils.preprocessing import preprocess

# Load dataset
(_, ds_val), ds_info = tfds.load('cats_vs_dogs', split=['train[:80%]', 'train[80%:]'], with_info=True, as_supervised=True)

# Apply the preprocessing function to the validation dataset
ds_val = ds_val.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

# Load the trained model
model = load_model('mobilenet_v2_cats_vs_dogs.h5')

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(ds_val)
print(f"Validation loss: {loss:.4f}")
print(f"Validation accuracy: {accuracy:.4f}")
