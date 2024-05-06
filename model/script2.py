import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import datasets, layers, models

# Define the ConvBlock class for convolutional layers followed by batch normalization and ReLU activation
class ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='same'):
        super(ConvBlock, self).__init__()
        self.conv = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Define the encoder network
def build_encoder():
    inputs = layers.Input(shape=(None, None, 4))  # Adjusted input shape to accommodate 4 channels
    x = ConvBlock(64, 5, strides=2)(inputs)
    x = ConvBlock(128, 5, strides=2)(x)
    x = ConvBlock(256, 3)(x)
    x = ConvBlock(256, 3)(x)
    x = ConvBlock(256, 3)(x)
    return models.Model(inputs, x, name='encoder')

# Define the decoder network
def build_decoder():
    inputs = layers.Input(shape=(None, None, 256))
    x = ConvBlock(256, 3)(inputs)
    x = ConvBlock(256, 3)(x)
    x = ConvBlock(128, 3)(x)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)
    x = ConvBlock(64, 3)(x)
    x = layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')(x)
    return models.Model(inputs, x, name='decoder')

# Define the complete DeepFill v2 model
def build_deepfill():
    encoder = build_encoder()
    decoder = build_decoder()

    inputs = layers.Input(shape=(None, None, 3))
    mask = layers.Input(shape=(None, None, 1))

    x = layers.Concatenate()([inputs, mask])
    x = encoder(x)
    x = decoder(x)

    # Apply mask to the output
    outputs = (1 - mask) * inputs + mask * x

    return models.Model([inputs, mask], outputs, name='deepfill')

# Function to generate random binary masks
def generate_random_mask(batch_size, height, width):
    masks = np.zeros((batch_size, height, width, 1))
    for i in range(batch_size):
        # Generate random coordinates for the top-left corner of the mask
        y = np.random.randint(0, height)
        x = np.random.randint(0, width)
        # Generate random width and height for the mask
        mask_height = np.random.randint(5, height // 2)
        mask_width = np.random.randint(5, width // 2)
        # Set the pixels within the randomly generated rectangle to 1
        masks[i, y:y+mask_height, x:x+mask_width] = 1
    return masks

# Function to load and preprocess images from folder
def load_and_preprocess_images(folder_path, target_size=(128, 128)):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            # Resize image to target size
            img = cv2.resize(img, target_size)
            # Convert image to float32 and normalize
            img = img.astype(np.float32) / 255.0
            # OpenCV loads images in BGR format, convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return np.array(images)

# Specify your folder path containing images
folder_path = "/content/images"

# Load and preprocess images from folder
train_images = load_and_preprocess_images(folder_path)

# Define DeepFill model
deepfill_model = build_deepfill()

# Define loss function
loss_fn = tf.keras.losses.MeanSquaredError()

# Define learning rate scheduler
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,  # Initial learning rate
    decay_steps=1000,  # Decay steps
    decay_rate=0.9)  # Decay rate

# Define optimizer with learning rate scheduling
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

# Training loop with learning rate scheduling
epochs = 1
batch_size = 32
total_iterations = 1000

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for iteration in range(total_iterations):
        i = np.random.randint(0, len(train_images) - batch_size)
        batch_images = train_images[i:i+batch_size]
        random_masks = generate_random_mask(len(batch_images), 128, 128)  # Adjusted mask size
        masked_images = batch_images * (1 - random_masks)

        with tf.GradientTape() as tape:
            inpainted_images = deepfill_model([masked_images, random_masks])
            loss = loss_fn(batch_images, inpainted_images)

        gradients = tape.gradient(loss, deepfill_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, deepfill_model.trainable_variables))
        print(f"Iteration {iteration}/{total_iterations} - Loss: {loss.numpy():.4f}")

print("Training finished.")

# Save the entire model
deepfill_model.save("model.save('back_project/model/deepfil.h5')")

