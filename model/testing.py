import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Load CIFAR-10 dataset with a smaller subset
(train_images, _), (test_images, _) = tf.keras.datasets.cifar100.load_data()
train_images = train_images[:1000]  # Selecting first 200 samples
test_images = test_images[:250]  # Selecting first 50 samples
print('Train images shape:', train_images.shape)
print('Train samples:', len(train_images))
print('Test samples:', len(test_images))

# Define data generator class with a smaller batch size
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, batch_size=8, image_size=(128, 128), n_channels=3, shuffle=True):
        self.batch_size = batch_size
        self.images = images
        self.image_size = image_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, index):
        idxs = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        return self.__data_generation(idxs)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


    def __data_generation(self, idxs):
        X_batch = np.empty((self.batch_size, *self.image_size, self.n_channels))
        y_batch = np.empty((self.batch_size, *self.image_size, self.n_channels))
        for i, idx in enumerate(idxs):
            x, y = np.random.randint(0, 21, 1)[0], np.random.randint(0, 21, 1)[0]  # Adjusted range for x and y
            w, h = np.random.randint(2, 5, 1)[0], np.random.randint(2, 5, 1)[0]  # Adjusted range for w and h (smaller)
            tmp_image = self.images[idx].copy()
            mask = np.full(tmp_image.shape, 255, np.uint8)
            mask[y:y+h, x:x+w] = 0
            res = cv2.bitwise_and(tmp_image, mask)
            X_batch[i] = cv2.resize(res / 255, self.image_size)  # Resize input images
            # Prepare ground truth images with missing regions already inpainted
            y_batch[i] = cv2.resize(tmp_image / 255, self.image_size)  # Resize original images
            y_batch[i][y:y+h, x:x+w] = X_batch[i][y:y+h, x:x+w]  # Inpaint missing regions
        return X_batch, y_batch


# Create data generators with a smaller batch size
train_generator = DataGenerator(train_images, batch_size=8, image_size=(128, 128))
test_generator = DataGenerator(test_images, batch_size=8, image_size=(128, 128))

# Define Dice coefficient function
def dice_coefficient(y_true, y_pred):
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), dtype=tf.float32)
    y_pred_f = tf.cast(tf.keras.backend.flatten(y_pred), dtype=tf.float32)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

# Define U-Net model
def create_unet_model():
    inputs = tf.keras.layers.Input((128, 128, 3))  # Adjusted input shape
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = tf.keras.layers.Conv2D(3, (1, 1), activation='sigmoid')(conv9)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[conv10])

    return model

# Create and compile model
model = create_unet_model()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=[dice_coefficient])
model.summary()

# Training loop
epochs = 5
dice_history = []
loss_history = []

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    dice_coefficients = []
    epoch_losses = []
    for batch_idx in range(len(train_generator)):
        X_batch, y_batch = train_generator[batch_idx]
        loss, dice = model.train_on_batch(X_batch, y_batch)
        epoch_losses.append(loss)
        dice_coefficients.append(dice)
        print(f'Batch {batch_idx + 1}/{len(train_generator)} - Loss: {loss:.4f} - Dice Coefficient: {dice:.4f}')

    avg_loss = np.mean(epoch_losses)
    avg_dice = np.mean(dice_coefficients)
    loss_history.append(avg_loss)
    dice_history.append(avg_dice)

    print(f'Average Loss: {avg_loss:.4f} - Average Dice Coefficient: {avg_dice:.4f}')

# Plot Dice coefficient across epochs
plt.plot(range(1, epochs + 1), dice_history, marker='o')
plt.title('Dice Coefficient Across Epochs')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.grid(True)
plt.show()

# Save the trained model
model.save('back_project/model/unet_model3.h5')
print("Model saved successfully.")

# Visualization of inpainted images
rows = 16
fig, axs = plt.subplots(nrows=rows, ncols=3, figsize=(20, 120))
for i in range(rows):
    # Generate a random index for the test set
    idx = np.random.randint(0, len(test_generator))
    sample_images, sample_labels = test_generator[idx]
    # Generate a random index for selecting an image from the sample
    img_idx = np.random.randint(0, len(sample_images) - 1)
    # Predict using the model
    inpainted_image = model.predict(sample_images[img_idx].reshape((1,) + sample_images[img_idx].shape))
    # Plot original image, ground truth, and inpainted image
    axs[i][0].imshow(sample_labels[img_idx])
    axs[i][0].set_title('Ground Truth')
    axs[i][1].imshow(sample_images[img_idx])
    axs[i][1].set_title('Original Image')
    axs[i][2].imshow(inpainted_image.reshape(inpainted_image.shape[1:]))
    axs[i][2].set_title('Inpainted Image')
plt.show()
