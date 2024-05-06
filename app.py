from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import io
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app, resources={r"/process_image": {"origins": "http://localhost:5173"}})

def dice_coefficient(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

model = tf.keras.models.load_model('./model/unet_model.h5', custom_objects={'dice_coefficient': dice_coefficient})

def inpaint_image(image, mask):
    # Extract alpha channel from the mask
    alpha_channel = mask[:, :, 3]

    # Convert alpha channel to binary mask
    binary_mask = (alpha_channel > 0).astype(np.uint8) * 255

    # Replicate the binary mask to match the number of channels in the image
    binary_mask = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)

    binary_mask = 255 - binary_mask
    # Apply the binary mask to the input image
    image_masked = image * (binary_mask / 255.0)

    # Resize the masked image to match the model's input shape
    image_resized = cv2.resize(image_masked, (32, 32))

    # Preprocess the resized image
    image_resized = image_resized.astype(np.float32) / 255.0

    # Predict the inpainted image
    inpainted_image = model.predict(np.expand_dims(image_resized, axis=0))

    # Post-process the predicted image
    inpainted_image = (inpainted_image.squeeze() * 255).astype(np.uint8)

    # Resize the inpainted image to match the original image dimensions
    inpainted_image_resized = cv2.resize(inpainted_image, (image.shape[1], image.shape[0]))

    return inpainted_image_resized


@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()
    # Extract dataURL and maskDataURL from JSON data
    dataURL = data.get('dataURL')
    maskDataURL = data.get('maskDataURL')

    # Check if 'dataURL' and 'maskDataURL' are present in the request
    if not dataURL or not maskDataURL:
        return jsonify({'error': 'Both image and mask dataURLs are required'})

    # Extract the base64-encoded image data from the dataURL
    image_data = dataURL.split(',')[1]
    image_binary = base64.b64decode(image_data)

    # Read the image using PIL and convert to RGB color space
    original_image = np.array(Image.open(io.BytesIO(image_binary)).convert('RGB'))

    # Decode the mask image from base64 and convert to PIL Image
    mask_data = base64.b64decode(maskDataURL.split(',')[1])
    mask_image = cv2.imdecode(np.frombuffer(mask_data, np.uint8), cv2.IMREAD_UNCHANGED)

    # Perform inpainting on the original image using the mask
    inpainted_image = inpaint_image(original_image, mask_image)

    # Combine the original image with the inpainted one using the mask
    combined_image = original_image.copy()
    combined_image[mask_image[:, :, 3] > 0] = inpainted_image[mask_image[:, :, 3] > 0]

    # Convert the combined image to PIL Image
    combined_pil = Image.fromarray(combined_image)


    # Plot the inpainted image and the mask
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(inpainted_image)
    axes[0].set_title('Inpainted Image')
    axes[0].axis('off')
    axes[1].imshow(mask_image, cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')
    plt.show()
    
    # Save the combined image to BytesIO object
    img_io = io.BytesIO()
    combined_pil.save(img_io, format='PNG')
    img_io.seek(0)

    # Return the combined image
    return send_file(img_io, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
