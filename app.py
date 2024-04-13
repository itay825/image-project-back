import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import io
import base64

app = Flask(__name__)
CORS(app)

# Define the custom metric function for dice coefficient
def dice_coefficient(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

# Register the custom metric function
tf.keras.utils.get_custom_objects()['dice_coefficient'] = dice_coefficient

# Load the model for inpainting
model = tf.keras.models.load_model('./model/unet_model.h5', custom_objects={'dice_coefficient': dice_coefficient})

def inpaint_image(image, model):
    # Ensure the image has only 3 channels (RGB)
    image = image[:, :, :3]
    # Resize the image to match the model's input shape
    image_resized = cv2.resize(image, (32, 32))
    # Preprocess the image
    image_resized = image_resized.astype(np.float32) / 255.0

    # Predict the inpainted image
    inpainted_image = model.predict(np.expand_dims(image_resized, axis=0))
    # Post-process the predicted image
    inpainted_image = (inpainted_image.squeeze() * 255).astype(np.uint8)
    return inpainted_image


def extract_image_and_mask(masked_image):
    # Convert masked image to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to get the mask
    ret, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Invert the mask
    mask = cv2.bitwise_not(mask)

    # Apply the mask to the original image
    image = cv2.bitwise_and(masked_image, masked_image, mask=mask)

    return image, mask

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()

    # Extract dataURL from JSON data
    dataURL = data.get('dataURL')

    # Check if 'dataURL' is present in the request
    if not dataURL:
        return jsonify({'error': 'No image provided'})

    # Extract the base64-encoded image data from the dataURL
    image_data = dataURL.split(',')[1]
    image_binary = base64.b64decode(image_data)

    # Read the image using PIL
    image = Image.open(io.BytesIO(image_binary))

    # Convert the image to a NumPy array for inpainting
    image_np = np.array(image)

    # Inpaint the image
    inpainted_image = inpaint_image(image_np, model)

    img, mask = extract_image_and_mask(image_np)

    # Convert the NumPy array back to an image
    result_image = Image.fromarray(inpainted_image)

    # Save the processed image to a BytesIO object
    img_io = io.BytesIO()
    result_image.save(img_io, 'PNG')
    img_io.seek(0)

    # Return the processed image
    return send_file(img_io, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
