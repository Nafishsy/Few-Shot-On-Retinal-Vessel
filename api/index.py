from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import os
from patchify import patchify, unpatchify
import base64
from io import BytesIO

# Global variables
PATCH_SIZE = 1024  # Reduced patch size for memory efficiency
BATCH_SIZE = 16
EPOCHS = 75
STRIDE = PATCH_SIZE // 2
STEP_SIZE = 50 * 4

# Initialize Flask app
app = Flask(__name__, template_folder='../templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Set the max upload size to 16MB

# Custom loss and metric functions (same as before)
@tf.keras.utils.register_keras_serializable()
def jacard_similarity(y_true, y_pred, epsilon=1e-7):
    y_true_f = tf.keras.backend.flatten(tf.keras.backend.cast(y_true, 'float32'))
    y_pred_f = tf.keras.backend.flatten(tf.keras.backend.cast(y_pred, 'float32'))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f + y_pred_f) - intersection
    return (intersection + epsilon) / (union + epsilon)
@tf.keras.utils.register_keras_serializable()
def jacard_loss(y_true, y_pred):
    return 1 - jacard_similarity(y_true, y_pred)
@tf.keras.utils.register_keras_serializable()
def dice_coefficient(y_true, y_pred, epsilon=1e-7):
    y_true_f = tf.keras.backend.flatten(tf.keras.backend.cast(y_true, 'float32'))
    y_pred_f = tf.keras.backend.flatten(tf.keras.backend.cast(y_pred, 'float32'))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + epsilon) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + epsilon)
@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)
@tf.keras.utils.register_keras_serializable()
def generalized_dice_coefficient(y_true, y_pred, epsilon=1e-7):
    y_true_f = tf.keras.backend.flatten(tf.keras.backend.cast(y_true, 'float32'))
    y_pred_f = tf.keras.backend.flatten(tf.keras.backend.cast(y_pred, 'float32'))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    weights = 1 / (tf.keras.backend.sum(y_true_f) ** 2 + epsilon)
    return (2.0 * intersection * weights + epsilon) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) * weights + epsilon)
@tf.keras.utils.register_keras_serializable()
def precision(y_true, y_pred, epsilon=1e-7):
    y_true_f = tf.keras.backend.flatten(tf.keras.backend.cast(y_true, 'float32'))
    y_pred_f = tf.keras.backend.flatten(tf.keras.backend.cast(y_pred, 'float32'))
    true_positives = tf.keras.backend.sum(y_true_f * y_pred_f)
    predicted_positives = tf.keras.backend.sum(y_pred_f)
    return (true_positives + epsilon) / (predicted_positives + epsilon)
@tf.keras.utils.register_keras_serializable()
def recall(y_true, y_pred, epsilon=1e-7):
    y_true_f = tf.keras.backend.flatten(tf.keras.backend.cast(y_true, 'float32'))
    y_pred_f = tf.keras.backend.flatten(tf.keras.backend.cast(y_pred, 'float32'))
    true_positives = tf.keras.backend.sum(y_true_f * y_pred_f)
    possible_positives = tf.keras.backend.sum(y_true_f)
    return (true_positives + epsilon) / (possible_positives + epsilon)
@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred):
    return jacard_loss(y_true, y_pred) + dice_loss(y_true, y_pred)
# Load the model
model_path = 'model/Model.keras'
model = load_model(
    model_path,
    custom_objects={  # Same custom objects
        "jacard_similarity": jacard_similarity,
        "jacard_loss": jacard_loss,
        "dice_coef": dice_coefficient,
        "dice_loss": dice_loss,
        "generalized_dice_coefficient": generalized_dice_coefficient,
        "precision": precision,
        "recall": recall,
        "combined_loss": combined_loss,
    },
)
# Define CLAHE object
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
# Preprocess the image
def preprocess_image(image):
    green_channel = image[:, :, 1]  # Extract green channel
    green_channel = clahe.apply(green_channel)
    return green_channel

# Efficient function to handle image prediction in patches
def predict_image(image):
    image_resized = cv2.resize(image, (2048, 2048))  # Resize image
    test_img = preprocess_image(image_resized)

    patches = patchify(test_img, (PATCH_SIZE, PATCH_SIZE), step=STRIDE)
    reconstructed_image = np.zeros_like(test_img)

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i, j, :, :]

            # Normalize the patch
            single_patch_norm = (single_patch.astype('float32')) / 255.0
            single_patch_input = np.expand_dims(np.expand_dims(single_patch_norm, axis=-1), 0)

            # Predict and place directly into reconstructed image
            single_patch_prediction = (model.predict(single_patch_input, verbose=0)[0, :, :, 0] > 0.5).astype(np.uint8)
            reconstructed_image[i * STRIDE:i * STRIDE + PATCH_SIZE,
                                j * STRIDE:j * STRIDE + PATCH_SIZE] = single_patch_prediction

    return reconstructed_image

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

from werkzeug.utils import secure_filename

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# Function to check allowed extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Load the uploaded image into memory without saving to disk
        uploaded_image = None
        if filename.lower().endswith(('.tif', '.tiff')):  # For TIFF files
            uploaded_image = np.array(Image.open(file))  # Open .tif using PIL
        else:  # For other image types like jpg/jpeg/png
            image_stream = np.asarray(bytearray(file.read()), dtype=np.uint8)
            uploaded_image = cv2.imdecode(image_stream, cv2.IMREAD_COLOR)  # Decode image

        # Store a copy of the uploaded image without any modifications
        original_image = uploaded_image.copy()

        # Predict the segmentation mask using the in-memory image
        prediction = predict_image(uploaded_image)

        # Render the images in the response
        img_bytes = BytesIO()

        # Plot the original and predicted segmentation together directly to BytesIO
        plt.figure(figsize=(10, 5))

        # Display the original image
        plt.subplot(1, 2, 1)
        if filename.lower().endswith(('.tif', '.tiff')):  # For TIFF files
            plt.imshow(original_image, cmap='gray')  # Show grayscale images for .tif or .tiff
        else:  # For RGB/BGR images
            plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct display
        plt.title("Original Image")
        plt.axis('off')

        # Display the prediction
        plt.subplot(1, 2, 2)
        plt.imshow(prediction, cmap='gray')  # Display segmentation mask with gray colormap
        plt.title("Predicted Segmentation")
        plt.axis('off')

        # Save the plot to the bytes buffer
        plt.savefig(img_bytes, format='png')
        plt.close()
        img_bytes.seek(0)

        # Prepare the image as a response
        response = {
            "original_image": "data:image/png;base64," + base64.b64encode(img_bytes.getvalue()).decode(),
            "result_image": "data:image/png;base64," + base64.b64encode(img_bytes.getvalue()).decode(),
        }
        return jsonify(response)

    return jsonify({"error": "Invalid file type"})

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Ensure correct port
    app.run(host='0.0.0.0', port=port)
