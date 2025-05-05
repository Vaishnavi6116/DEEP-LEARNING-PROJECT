# DEEP-LEARNING-PROJECT
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------- LOAD DATA ----------------------
def load_data():
    print("Loading Fashion MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
    return (x_train, y_train), (x_test, y_test)

# ---------------------- BUILD MODEL ----------------------
def build_model():
    print("Building model...")
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---------------------- TRAIN MODEL ----------------------
def train_model(model, x_train, y_train, x_test, y_test):
    print("Training model...")
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    return history

# ---------------------- PLOT RESULTS ----------------------
def plot_history(history):
    print("Plotting training history...")
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# ---------------------- PREDICT SAMPLE IMAGES ----------------------
def show_predictions(model, x_test, y_test, class_names):
    print("Showing predictions on sample test images...")
    predictions = model.predict(x_test)
    plt.figure(figsize=(10, 10))

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        # Resize the image for better visual quality (from 28x28 to 112x112)
        resized_image = tf.image.resize(x_test[i][..., np.newaxis], [112, 112])
        plt.imshow(tf.squeeze(resized_image), cmap='gray', interpolation='bilinear')

        pred_label = np.argmax(predictions[i])
        true_label = y_test[i]
        color = 'green' if pred_label == true_label else 'red'
        plt.xlabel(f"{class_names[pred_label]} ({class_names[true_label]})", color=color)

    plt.tight_layout()
    plt.show()

# ---------------------- MAIN ----------------------
def run():
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()
    history = train_model(model, x_train, y_train, x_test, y_test)
    plot_history(history)
    show_predictions(model, x_test, y_test, class_names)

#Corrected the condition to use __name__
if __name__ == "__main__":
    run()
