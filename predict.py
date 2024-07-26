import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils.image_utils import load_and_preprocess_image

# Path to your image
img_path = 'example.jpg'

# Load and preprocess the image
original_img, img_array = load_and_preprocess_image(img_path)

# Load the trained model
model = load_model('mobilenet_v2_cats_vs_dogs.h5')

# Predict the class of the image
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]

# Map the predicted class to the actual class name
class_names = ['cat', 'dog']
predicted_class_name = class_names[predicted_class]

# Print the predicted class
print(f'Predicted class: {predicted_class_name}')

# Display the image with the predicted class
plt.figure(figsize=(6, 6))
plt.imshow(original_img)
plt.title(f'Predicted class: {predicted_class_name}')
plt.axis('off')
plt.show()
