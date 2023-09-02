# Use a breakpoint in the code line below to debug your script.
# Press Ctrl+F8 to toggle the breakpoint.

import tensorflow as tf

def import_data():

    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Normalize the pixel values from a range of [0, 255] to [0, 1]
    #train_images = train_images / 255.0
    #test_images = test_images / 255.0

    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    return (train_images, train_labels), (test_images, test_labels)

def check_data_shape():
    # If you want to check the shape of the datasets
    print(train_images.shape)  # Should print (60000, 28, 28)
    print(train_labels.shape)  # Should print (60000,)
    print(test_images.shape)  # Should print (10000, 28, 28)
    print(test_labels.shape)  # Should print (10000,)

def set_up_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (2, 2), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (2, 2), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(16, (2, 2), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(196, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='RMSprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_images, train_labels, num_epochs):
    model.fit(train_images, train_labels, epochs=num_epochs)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = import_data()
    model = set_up_model()
    train_model(model, train_images, train_labels, 25)


