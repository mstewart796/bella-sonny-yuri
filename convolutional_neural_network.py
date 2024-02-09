import tensorflow as tf
import scipy
from keras.preprocessing.image import ImageDataGenerator

# Define the directories for training and testing data
train_dir = 'dataset/training_set'
test_dir = 'dataset/test_set'

# Set up data generators
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='categorical')

# Define the model
cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=3, activation='softmax')  # 3 units for 3 classes
])

# Compile the model
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

# Make a prediction
import numpy as np
from keras.preprocessing import image

# Load and preprocess the image
test_image = image.load_img('dataset/single_prediction/bella_sonny_or_yuri.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Predict the result
result = cnn.predict(test_image)
class_indices = training_set.class_indices

# Interpret the prediction
predicted_class = list(class_indices.keys())[np.argmax(result)]

print("Predicted class:", predicted_class)
