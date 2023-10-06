import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
def load_cats_vs_dogs_data():
    train_data = tfds.load('cats_vs_dogs', split='train[:80%]', as_supervised=True)
    valid_data = tfds.load('cats_vs_dogs', split='train[80%:90%]', as_supervised=True)
    test_data = tfds.load('cats_vs_dogs', split='train[-10%:]', as_supervised=True)

    return train_data, valid_data, test_data

def data_to_lists(data, img_width=120, img_height=120):
    images = []
    labels = []

    for image, label in data:
        resized_image = tf.image.resize(image, [img_width, img_height]).numpy()
        images.append(resized_image)
        labels.append(label.numpy())

    return images, labels

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(120, 120, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

BATCH_SIZE = 32
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_data, valid_data, test_data = load_cats_vs_dogs_data()
    (train_images, train_labels) = data_to_lists(train_data)
    (validation_images, validation_labels) = data_to_lists(valid_data)
    (test_images, test_labels) = data_to_lists(test_data)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_gen = train_datagen.flow(np.array(train_images), np.array(train_labels), batch_size=BATCH_SIZE)
    validation_gen = train_datagen.flow(np.array(validation_images), np.array(validation_labels), batch_size=BATCH_SIZE)
    model = create_model()
    model.fit(train_gen, validation_data=validation_gen, epochs=30, steps_per_epoch=len(train_images))

