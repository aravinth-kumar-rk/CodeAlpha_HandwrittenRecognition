import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# 1. Load the MNIST handwritten digits dataset
print("Loading MNIST dataset...")
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Reshape images to fit the CNN input requirement
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 2. Build the CNN using the modern Input layer syntax
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),  # Modern syntax replaces input_shape in Conv2D
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 3. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the model
print("\nTraining CNN Model...")
history = model.fit(train_images, train_labels, epochs=3, validation_data=(test_images, test_labels))

# 5. Evaluate final performance
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nHandwritten Character Recognition Accuracy: {test_acc * 100:.2f}%")
