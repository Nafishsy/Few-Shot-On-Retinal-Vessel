from tensorflow.keras import backend as K
import tensorflow as tf

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
    return dice_coef_loss(y_true, y_pred) + jaccard_loss(y_true, y_pred)
