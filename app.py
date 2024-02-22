from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import io
import base64

app = Flask(__name__)
CORS(app)

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()

    # Extract operation, dataURL, and scaleValue from JSON data
    operation = data.get('operation')
    dataURL = data.get('dataURL')
    scaleValue = data.get('scaleValue')  # Add this line

    # Check if 'dataURL' is present in the request
    if not dataURL:
        return jsonify({'error': 'No image provided'})

    # Extract the base64-encoded image data from the dataURL
    image_data = dataURL.split(',')[1]
    image_binary = base64.b64decode(image_data)

    # Read the image using PIL
    image = Image.open(io.BytesIO(image_binary))

    # Convert the image to a NumPy array for OpenCV processing
    image_np = np.array(image)

    # Image processing logic
    if operation == 'Image Sharpening':
        sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        image_np = cv2.filter2D(image_np, -1, sharpening_kernel)
    elif operation == 'Image Blurring':
        blur_size = (5, 5)
        image_np = cv2.blur(image_np, blur_size)
    elif operation == 'Color Grayscale':
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    elif operation == 'Color Inversion':
        image_np = cv2.bitwise_not(image_np)
    elif operation == 'Rotation':
        angle = 90
        rows, cols, _ = image_np.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        image_np = cv2.warpAffine(image_np, rotation_matrix, (cols, rows))
    elif operation == 'Scaling':
        # Use the scaleValue received from the frontend
        scale_factor = scaleValue
        image_np = cv2.resize(image_np, None, fx=scale_factor, fy=scale_factor)
    elif operation == 'Binary Thresholding':
        _, image_np = cv2.threshold(cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    elif operation == 'Adaptive Thresholding':
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        image_np = cv2.adaptiveThreshold(image_np, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        return jsonify({'error': 'Invalid operation'})

    # Convert the NumPy array back to an image
    result_image = Image.fromarray(image_np)

    # Save the processed image to a BytesIO object
    img_io = io.BytesIO()
    result_image.save(img_io, 'PNG')
    img_io.seek(0)

    # Return the processed image
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
