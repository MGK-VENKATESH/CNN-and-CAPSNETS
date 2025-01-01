import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import os

# Dataset path
data_dir = '/Users/mgkvenky/Desktopmgkvenky# Define the CNN model
#def CNN(input_shape, n_class):


def CNN(input_shape, n_class):
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.01)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        layers.Dropout(0.3),
        layers.Dense(n_class, activation='softmax')
    ])
    return model

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
class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
class_weight_dict = dict(enumerate(class_weights))

# Build the model
input_shape = (64, 64, 1)  # Adjust this based on your image size
n_class = 4  # Number of classes (cloudy, desert, green_area, water)

model = CNN(input_shape, n_class)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Initialize lists to store accuracies
train_accuracies = []
val_accuracies = []
test_accuracies = []

# Training loop
epochs = 50
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=1,
        class_weight=class_weight_dict
    )
    
    # Store training and validation accuracies
    train_accuracies.append(history.history['accuracy'][0])
    val_accuracies.append(history.history['val_accuracy'][0])
    
    # Evaluate on test set after each epoch
    _, test_accuracy = model.evaluate(test_generator)
    test_accuracies.append(test_accuracy)
    
    print(f"Train Accuracy: {train_accuracies[-1]:.4f}")
    print(f"Validation Accuracy: {val_accuracies[-1]:.4f}")
    print(f"Test Accuracy: {test_accuracies[-1]:.4f}")

# Plot the accuracies
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, epochs+1), val_accuracies, label='Validation Accuracy')
plt.plot(range(1, epochs+1), test_accuracies, label='Test Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png')
plt.close()

# Final evaluation on test set
final_test_loss, final_test_accuracy = model.evaluate(test_generator)
print(f"Final Test Accuracy: {final_test_accuracy:.4f}")

# Get predictions for the test set
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes[:len(y_pred_classes)]

# Print classification report and confusion matrix for the test set
print("\nTest Set Results:")
print(classification_report(y_true, y_pred_classes, target_names=list(test_generator.class_indices.keys())))
print(confusion_matrix(y_true, y_pred_classes))
