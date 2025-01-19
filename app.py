from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from PIL import Image  # Importing PIL for handling TIFF files
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify

# Global variables
PATCH_SIZE = 2048 // 2  # Updated PATCH_SIZE
BATCH_SIZE = 16
EPOCHS = 75
STRIDE = PATCH_SIZE // 2
STEP_SIZE = 50 * 4

# Initialize Flask app with static folder for uploads
app = Flask(__name__, static_url_path='/uploads', static_folder='uploads')

# Set the path for the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}  # Added 'tif' and 'tiff'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure 'uploads' folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Custom metrics and loss functions
from tensorflow.keras import backend as K

@tf.keras.utils.register_keras_serializable()
def jacard_similarity(y_true, y_pred, epsilon=1e-7):
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f) - intersection
    return (intersection + epsilon) / (union + epsilon)

@tf.keras.utils.register_keras_serializable()
def jaccard_loss(y_true, y_pred):
    return 1 - jacard_similarity(y_true, y_pred)

@tf.keras.utils.register_keras_serializable()
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

@tf.keras.utils.register_keras_serializable()
def generalized_dice_coefficient(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

@tf.keras.utils.register_keras_serializable()
def dice_coef_loss(y_true, y_pred):
    return 1 - generalized_dice_coefficient(y_true, y_pred)

@tf.keras.utils.register_keras_serializable()
def accuracy(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.round(K.flatten(y_pred))
    correct = K.equal(y_true_f, y_pred_f)
    return K.mean(K.cast(correct, dtype='float32'))

@tf.keras.utils.register_keras_serializable()
def precision(y_true, y_pred):
    y_true_f = K.cast(K.flatten(y_true), dtype='float32')
    y_pred_f = K.round(K.cast(K.flatten(y_pred), dtype='float32'))
    true_positives = K.sum(y_true_f * y_pred_f)
    predicted_positives = K.sum(y_pred_f)
    return (true_positives + 1.0) / (predicted_positives + 1.0)

@tf.keras.utils.register_keras_serializable()
def recall(y_true, y_pred):
    y_true_f = K.cast(K.flatten(y_true), dtype='float32')
    y_pred_f = K.round(K.cast(K.flatten(y_pred), dtype='float32'))
    true_positives = K.sum(y_true_f * y_pred_f)
    possible_positives = K.sum(y_true_f)
    return (true_positives + 1.0) / (possible_positives + 1.0)

@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred):
    loss1 = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    loss2 = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return loss1 + loss2


# Load the model
model_path = 'model/Model.keras'  # Change to your actual .keras model file path
model = load_model(
    model_path,
    custom_objects={
        "combined_loss": combined_loss,
        "jacard_similarity": jacard_similarity,
        "jaccard_loss": jaccard_loss,
        "dice_coef": dice_coef,
        "generalized_dice_coefficient": generalized_dice_coefficient,
        "dice_coef_loss": dice_coef_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    },
)

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define CLAHE object
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# Define the image preprocessing function with CLAHE on the green channel
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    green_channel = image[:, :, 1]
    green_channel = clahe.apply(green_channel)
    return green_channel

# Define a function to predict using the model
def predict_image(image):
    image_resized = cv2.resize(image, (2048, 2048))
    test_img = preprocess_image(image_resized)

    patches = patchify(test_img, (PATCH_SIZE, PATCH_SIZE), step=STRIDE)
    predicted_patches = []
    
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i, j, :, :]
            single_patch_norm = (single_patch.astype('float32')) / 255.0
            single_patch_input = np.expand_dims(np.expand_dims(single_patch_norm, axis=-1), 0)
            single_patch_prediction = (model.predict(single_patch_input, verbose=0)[0, :, :, 0] > 0.5).astype(np.uint8)
            predicted_patches.append(single_patch_prediction)
    
    predicted_patches = np.array(predicted_patches)
    predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], PATCH_SIZE, PATCH_SIZE))
    reconstructed_image = unpatchify(predicted_patches_reshaped, test_img.shape)
    return reconstructed_image

# Home route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Save the original image as is, without any modification
        original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_saved.png')
        if filename.lower().endswith(('.tif', '.tiff')):  # For TIFF files
            image = np.array(Image.open(filepath))  # Open .tif using PIL
            Image.fromarray(image).save(original_image_path)  # Save as is for TIFF
        else:  # For other image types like jpg/jpeg/png
            image = cv2.imread(filepath)  # Open with OpenCV (BGR)
            cv2.imwrite(original_image_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Save in RGB format

        # Predict the segmentation mask using a copy of the image
        prediction = predict_image(image)

        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.png')
        plt.figure(figsize=(10, 5))

        # Display the original image using the saved version
        original_saved_image = cv2.imread(original_image_path)
        plt.subplot(1, 2, 1)
        plt.imshow(original_saved_image)  # Render the saved image in RGB
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(prediction, cmap='gray')
        plt.title("Predicted Segmentation")
        plt.axis('off')

        plt.savefig(result_image_path)
        plt.close()

        # Prepare the result image URLs
        original_image_url = f'/uploads/original_saved.png'
        result_image_url = f'/uploads/result.png'

        return jsonify({"original_image": original_image_url, "result_image": result_image_url})

    return jsonify({"error": "Invalid file type"})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
