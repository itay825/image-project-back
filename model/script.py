import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Define the custom metric function
def dice_coefficient(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

model = tf.keras.models.load_model('deepfil.h5', custom_objects={'dice_coefficient': dice_coefficient})

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
    inpainted_image = cv2.resize(inpainted_image, (500, 500), Image.ANTIALIAS)
    return inpainted_image, mask, image_resized_masked, image_resized

# Load an example image
example_image = cv2.imread('123.png')
height, width, _ = example_image.shape
print("Example Image Width:", width)
print("Example Image Height:", height)

# Perform inpainting
inpainted_example, mask, image_resized_masked, image_resized = inpaint_image(example_image)

# Resize the mask to match the original image size
resized_mask = cv2.resize(mask, (width, height))

# Resize the inpainted image to match the original image size
resized_inpainted = cv2.resize(inpainted_example, (width, height))

# Combine the inpainted image with the original image using the resized mask
combined_image = np.copy(example_image)
combined_image[resized_mask == 0] = resized_inpainted[resized_mask == 0]

# Display original image, resized image with mask, mask, inpainted image, and combined image
plt.figure(figsize=(20, 10))

# Original Image
plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Resized Image
plt.subplot(2, 3, 2)
plt.title('Resized Image')
plt.imshow(cv2.cvtColor((image_resized * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.axis('off')

# Resized Image with Mask
plt.subplot(2, 3, 3)
plt.title('Resized Image with Mask')
plt.imshow(cv2.cvtColor((image_resized_masked * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.axis('off')

# Mask
plt.subplot(2, 3, 4)
plt.title('Mask')
plt.imshow(resized_mask, cmap='gray')
plt.axis('off')

# Inpainted Image
plt.subplot(2, 3, 5)
plt.title('Inpainted Image')
plt.imshow(cv2.cvtColor(inpainted_example, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Resized Image with Mask and Inpainted Image combined
plt.subplot(2, 3, 6)
plt.title('Inpainted Image on Original')
plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()


# Save the combined image
# cv2.imwrite('combined_image.png', combined_image)
