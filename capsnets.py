import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# Dataset path
data_dir = '/Users/mgkvenky/Desktop/data'

# Define the squash activation function
def squash(x, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(x), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())
    return scale * x

# Define the CapsNet model
def CapsNet(input_shape, n_class, n_route):
    x = layers.Input(shape=input_shape)
    
    # Layer 1: Conv2D layer
    conv1 = layers.Conv2D(256, 9, activation='relu')(x)
    
    # Layer 2: PrimaryCaps layer
    primarycaps = layers.Conv2D(32, 9, strides=2, padding='valid', activation='relu')(conv1)
    primarycaps = layers.Reshape((-1, 8))(primarycaps)
    primarycaps = layers.Lambda(squash)(primarycaps)
    
    # Layer 3: DigitCaps layer
    digitcaps = layers.Dense(16, activation=None)(primarycaps)
    digitcaps = layers.Lambda(squash)(digitcaps)
    
    # Length layer
    out_caps = layers.Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.square(x), -1)))(digitcaps)
    
    # Add a Dense layer to get the correct number of outputs
    outputs = layers.Dense(n_class, activation='softmax')(out_caps)
    
    return models.Model(x, outputs)

# Load and preprocess the data
img_size = (64, 64)  # Adjust this based on your image size
batch_size = 32

# Add data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Split the data into train, validation, and test sets
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    color_mode='grayscale',
    seed=42
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    color_mode='grayscale',
    seed=42
)

# Create a separate test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)

# Calculate class weights
# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
class_weight_dict = dict(enumerate(class_weights))


# Modify the margin loss function to include class weights
def weighted_margin_loss(y_true, y_pred):
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
    weights = tf.reduce_sum(y_true * tf.constant(list(class_weight_dict.values()), dtype=tf.float32), axis=1)
    return tf.reduce_mean(tf.reduce_sum(L, axis=1) * weights)

# Build the model
input_shape = (64, 64, 1)  # Adjust this based on your image size
n_class = 4  # Number of classes (cloudy, desert, green_area, water)
n_route = 3  # Number of routing iterations

model = CapsNet(input_shape, n_class, n_route)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Custom training loop
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = weighted_margin_loss(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions

# Training loop
epochs = 50
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    total_loss = 0
    num_batches = 0
    for x_batch, y_batch in train_generator:
        loss, predictions = train_step(x_batch, y_batch)
        total_loss += loss
        num_batches += 1
        if num_batches >= train_generator.samples // batch_size:
            break
    
    avg_loss = total_loss / num_batches
    
    # Validation
    val_losses = []
    val_accuracies = []
    num_val_batches = 0
    for x_val, y_val in validation_generator:
        val_predictions = model(x_val)
        val_loss = weighted_margin_loss(y_val, val_predictions)
        val_accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_val, val_predictions))
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        num_val_batches += 1
        if num_val_batches >= validation_generator.samples // batch_size:
            break
    
    print(f"Training loss: {avg_loss:.4f}")
    print(f"Validation loss: {tf.reduce_mean(val_losses):.4f}")
    print(f"Validation accuracy: {tf.reduce_mean(val_accuracies):.4f}")

# Evaluate the model on the validation set
y_pred_val = []
y_true_val = []

num_eval_batches = 0
for x_val, y_val in validation_generator:
    batch_pred = model(x_val)
    y_pred_val.extend(tf.argmax(batch_pred, axis=1).numpy())
    y_true_val.extend(tf.argmax(y_val, axis=1).numpy())
    num_eval_batches += 1
    if num_eval_batches * batch_size >= validation_generator.samples:
        break

y_pred_val = np.array(y_pred_val)
y_true_val = np.array(y_true_val)

# Ensure we have the same number of predictions as true labels
min_len = min(len(y_pred_val), len(y_true_val))
y_pred_val = y_pred_val[:min_len]
y_true_val = y_true_val[:min_len]

print("Validation Set Results:")
print(classification_report(y_true_val, y_pred_val, target_names=list(validation_generator.class_indices.keys())))
print(confusion_matrix(y_true_val, y_pred_val))
val_accuracy = np.mean(y_true_val == y_pred_val)
print(f"Validation accuracy: {val_accuracy:.4f}")

# Evaluate the model on the test set
y_pred_test = []
y_true_test = []

num_test_batches = 0
for x_test, y_test in test_generator:
    batch_pred = model(x_test)
    y_pred_test.extend(tf.argmax(batch_pred, axis=1).numpy())
    y_true_test.extend(tf.argmax(y_test, axis=1).numpy())
    num_test_batches += 1
    if num_test_batches * batch_size >= test_generator.samples:
        break

y_pred_test = np.array(y_pred_test)
y_true_test = np.array(y_true_test)

# Ensure we have the same number of predictions as true labels
min_len = min(len(y_pred_test), len(y_true_test))
y_pred_test = y_pred_test[:min_len]
y_true_test = y_true_test[:min_len]

print("\nTest Set Results:")
print(classification_report(y_true_test, y_pred_test, target_names=list(test_generator.class_indices.keys())))
print(confusion_matrix(y_true_test, y_pred_test))
test_accuracy = np.mean(y_true_test == y_pred_test)
print(f"Test accuracy: {test_accuracy:.4f}")
