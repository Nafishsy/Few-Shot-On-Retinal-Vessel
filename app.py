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
from patchify import patchify
import base64
from io import BytesIO
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Flask app setup
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # Max upload size (64 MB)

# GPU memory growth setting
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Constants
PATCH_SIZE = 1024
STRIDE = PATCH_SIZE // 2
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

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

# CLAHE for preprocessing
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


# Utility: Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Preprocess image (extract green channel + CLAHE)
def preprocess_image(image):
    green_channel = image[:, :, 1]  # Extract green channel
    return clahe.apply(green_channel)


# Predict segmentation using patches
def predict_image(image):
    image_resized = cv2.resize(image, (2048, 2048))
    processed_image = preprocess_image(image_resized)

    patches = patchify(processed_image, (PATCH_SIZE, PATCH_SIZE), step=STRIDE)
    reconstructed_image = np.zeros_like(processed_image)

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i, j, :, :]
            single_patch_norm = single_patch.astype('float32') / 255.0
            single_patch_input = np.expand_dims(np.expand_dims(single_patch_norm, axis=-1), 0)

            # Predict and update reconstructed image
            prediction_patch = (model.predict(single_patch_input, verbose=0)[0, :, :, 0] > 0.5).astype(np.uint8)
            reconstructed_image[i * STRIDE:i * STRIDE + PATCH_SIZE, 
                                j * STRIDE:j * STRIDE + PATCH_SIZE] = prediction_patch

    return reconstructed_image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"})

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"})

        if not allowed_file(file.filename):
            return jsonify({"error": "Unsupported file type"})

        # Load image from request
        filename = secure_filename(file.filename)
        if filename.lower().endswith(('.tif', '.tiff')):
            uploaded_image = np.array(Image.open(file))
        else:
            image_stream = np.asarray(bytearray(file.read()), dtype=np.uint8)
            uploaded_image = cv2.imdecode(image_stream, cv2.IMREAD_COLOR)

        # Store a copy of the uploaded image
        original_image = uploaded_image.copy()

        # Predict the segmentation mask
        prediction = predict_image(uploaded_image)
        if prediction is None or prediction.size == 0:
            return jsonify({"error": "Prediction failed: Empty or invalid output."})

        # Create visualization
        img_bytes = BytesIO()
        plt.figure(figsize=(10, 5))

        # Original image
        plt.subplot(1, 2, 1)
        if filename.lower().endswith(('.tif', '.tiff')):
            plt.imshow(original_image, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')

        # Predicted segmentation
        plt.subplot(1, 2, 2)
        plt.imshow(prediction, cmap='gray')
        plt.title("Predicted Segmentation")
        plt.axis('off')

        # Save plot to buffer
        plt.savefig(img_bytes, format='png')
        plt.close()
        img_bytes.seek(0)

        # Prepare response with base64-encoded images
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode()
        return jsonify({
            "original_image": f"data:image/png;base64,{img_base64}",
            "result_image": f"data:image/png;base64,{img_base64}",
        })

    except Exception as e:
        print(f"Error in prediction route: {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"})


# Run Flask app
if __name__ == '__main__':
    try:
        port = int(os.environ.get("PORT", 10000))
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        print(f"Error starting the server: {e}")
