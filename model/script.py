import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define the custom metric function
def dice_coefficient(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

# Load the saved model with custom_objects argument
model = tf.keras.models.load_model('model/unet_model.h5', custom_objects={'dice_coefficient': dice_coefficient})

# Function to perform image inpainting using the loaded model
def inpaint_image(image):
    # Resize the image to match the model's input shape
    image_resized = cv2.resize(image, (32, 32))
    # Preprocess the image
    image_resized = image_resized.astype(np.float32) / 255.0

    # Create a smaller rectangular mask
    mask = np.zeros_like(image_resized)
    mask[12:20, 12:20, :] = 1  # Define the rectangle position and size

    # Invert the mask
    mask = 1 - mask

    # Apply the mask to the input image
    image_resized_masked = image_resized * mask

    # Predict the inpainted image
    inpainted_image = model.predict(np.expand_dims(image_resized_masked, axis=0))
    # Post-process the predicted image
    inpainted_image = (inpainted_image.squeeze() * 255).astype(np.uint8)
    return inpainted_image, mask, image_resized_masked, image_resized



# Example usage:
# Load an example image
example_image = cv2.imread('123.png')  # Replace '123.png' with your image path
# Perform inpainting
inpainted_example, mask, image_resized_masked, image_resized = inpaint_image(example_image)

# Display original image, resized image with mask, mask, and inpainted image
plt.figure(figsize=(20, 5))

# Original Image
plt.subplot(1, 5, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Resized Image
plt.subplot(1, 5, 2)
plt.title('Resized Image')
plt.imshow(cv2.cvtColor((image_resized * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.axis('off')

# Resized Image with Mask
plt.subplot(1, 5, 3)
plt.title('Resized Image with Mask')
plt.imshow(cv2.cvtColor((image_resized_masked * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.axis('off')

# Mask
plt.subplot(1, 5, 4)
plt.title('Mask')
plt.imshow(mask, cmap='gray')
plt.axis('off')

# Inpainted Image
plt.subplot(1, 5, 5)
plt.title('Inpainted Image')
plt.imshow(cv2.cvtColor(inpainted_example, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
