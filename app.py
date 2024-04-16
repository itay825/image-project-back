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

def inpaint_image(image, startX, startY, endX, endY):
    print("Input image shape:", image.shape)

    margin=5
    startX = max(0, startX - margin)
    startY = max(0, startY - margin)
    endX = min(image.shape[1], endX + margin)
    endY = min(image.shape[0], endY + margin)

    print("Top-right corner coordinates:", (endX, startY))
    print("Bottom-left corner coordinates:", (startX, endY))
    
    # Create a mask based on the provided coordinates
    mask = np.ones_like(image)
    mask[startY:endY, startX:endX, :] = 0
    
    # Apply the mask to the input image
    image_masked = image * mask

    # Resize the masked image to match the model's input shape
    image_resized = cv2.resize(image_masked, (32, 32))

    # Preprocess the resized image
    image_resized = image_resized.astype(np.float32) / 255.0

    # Predict the inpainted image
    inpainted_image = model.predict(np.expand_dims(image_resized, axis=0))

    # Post-process the predicted image
    inpainted_image = (inpainted_image.squeeze() * 255).astype(np.uint8)
    
    return inpainted_image, mask, image_resized, image_masked


@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()
    # Extract dataURL and coordinates from JSON data
    dataURL = data.get('dataURL')
    startX = int(data.get('startX'))
    startY = int(data.get('startY'))
    endX = int(data.get('endX'))
    endY = int(data.get('endY'))
    print("cords", startX, startY, endX, endY)

    # Check if 'dataURL' is present in the request
    if not dataURL:
        return jsonify({'error': 'No image provided'})

    # Extract the base64-encoded image data from the dataURL
    image_data = dataURL.split(',')[1]
    image_binary = base64.b64decode(image_data)

    # Read the image using PIL
    image_np = np.frombuffer(image_binary, np.uint8)
    example_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Inpaint the image with adjusted coordinates
    inpainted_example, mask, image_resized_masked, image_resized = inpaint_image(example_image, startX, startY, endX, endY)

    # Resize the mask and inpainted image to original size
    resized_mask = cv2.resize(mask, (example_image.shape[1], example_image.shape[0]))
    resized_inpainted = cv2.resize(inpainted_example, (example_image.shape[1], example_image.shape[0]))
    combined_image = np.copy(example_image)
    combined_image[resized_mask == 0] = resized_inpainted[resized_mask == 0]

    # Save the processed image to a BytesIO object
    img_io = io.BytesIO()
    combined_image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
    combined_image_pil = Image.fromarray(combined_image_rgb)
    combined_image_pil.save(img_io, 'PNG')

    img_io.seek(0)

    # Return the processed image
    return send_file(img_io, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
