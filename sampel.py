import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, SpatialDropout2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Load and preprocess data
(x_train, labels_train), (x_test, labels_test) = mnist.load_data()

# Convert to float and normalize
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert labels to categorical
y_train = to_categorical(labels_train, 10)
y_test = to_categorical(labels_test, 10)

# Reshape data for CNN (remove the redundant reshape to 784)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Create proper validation split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                 test_size=0.2, 
                                                 random_state=42)

# Data augmentation setup
datagen = ImageDataGenerator(
    rotation_range=8,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

# Define the improved model architecture
net = Sequential([
    # First Convolutional Block
    Conv2D(32, (5,5), activation='relu', padding='same', input_shape=(28,28,1)),
    BatchNormalization(),
    Conv2D(32, (5,5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),  # No overlapping pooling (stride = pool_size)
    SpatialDropout2D(0.25),
    
    # Second Convolutional Block
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),  # No overlapping pooling (stride = pool_size)
    SpatialDropout2D(0.25),
    
    # Third Convolutional Block
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),  # No overlapping pooling (stride = pool_size)
    SpatialDropout2D(0.25),
    
    # Dense Layers
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile model
net.compile(loss='categorical_crossentropy',
           optimizer='adam',
           metrics=['accuracy'])

# Define callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                             factor=0.2,
                             patience=3,
                             min_lr=1e-6,
                             verbose=1)

early_stop = EarlyStopping(monitor='val_loss',
                          patience=5,
                          restore_best_weights=True,
                          verbose=1)

# Train the model using data augmentation
history = net.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    validation_data=(x_val, y_val),
    epochs=30,
    callbacks=[reduce_lr, early_stop]
)

# Evaluate on test set
test_loss, test_accuracy = net.evaluate(x_test, y_test, verbose=0)
print(f'Test accuracy: {test_accuracy:.4f}')

# Save the model
net.save('mnist_model.h5')

# Plotting training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Confusion Matrix and Classification Report
from sklearn.metrics import confusion_matrix, classification_report
predictions = net.predict(x_test)
pred_labels = np.argmax(predictions, axis=1)
true_labels = labels_test

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels))
