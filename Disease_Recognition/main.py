import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
def main():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Preprocess the data
    train_images = train_images.reshape((-1, 28, 28, 1)) / 255.0
    test_images = test_images.reshape((-1, 28, 28, 1)) / 255.0
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # Build the CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=5, batch_size=128, validation_data=(test_images, test_labels))

    # Choose a random test image
    index = np.random.randint(0, len(test_images))
    image = test_images[index]
    label = np.argmax(test_labels[index])

    # Perform prediction on the test image
    prediction = model.predict(np.expand_dims(image, axis=0))
    predicted_label = np.argmax(prediction)

    # Display the test image and prediction result
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"True Label: {label}, Predicted Label: {predicted_label}")
    plt.axis('off')
    plt.show()
