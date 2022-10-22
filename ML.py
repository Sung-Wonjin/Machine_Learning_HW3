import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

base_dir = r'venv/kaggle/input/Multi-class Weather Dataset'
folders = os.listdir(base_dir)
print(folders)
#load data and split into train data and validaion data
train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) #set validation split

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(250, 250),
    batch_size= 30,
    class_mode='categorical',
    subset='training')# set as training data

validation_generator = train_datagen.flow_from_directory(
    base_dir, # same directory as training data
    target_size=(250, 250),
    batch_size= 30,
    class_mode='categorical',
    subset='validation') # set as validation data

#define labels and number of classes to classify
labels = (train_generator.class_indices)
labels = dict((v , k) for k , v in labels.items())
print(labels)
num_classes = 4

with tf.device('/device:GPU:0'):
    input = layers.Input(shape=(250, 250, 3))
    #define layers
    x = layers.Conv2D(32, (5, 5), activation='relu', input_shape=(250, 250, 3), padding='same')(input)
    x = layers.MaxPooling2D((3, 3), padding='same')(x)
    x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((3, 3), padding='same')(x)
    x = layers.Conv2D(128, (5, 5), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((3, 3), padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    #define model and fit the model into dataset
    model = Model(input, x)
    model.summary()
    model.compile(optimizer='adam', loss=losses.CategoricalCrossentropy(), metrics=['accuracy'])
    model.fit(train_generator, validation_data=validation_generator, epochs=15)
    predictions = model.predict(validation_generator)
    test_loss, test_acc = model.evaluate(validation_generator, verbose=2)
    print(test_acc)

    test_path = r'venv/kaggle/input/test/rain1.jpg'
    test_image = tf.keras.preprocessing.image.load_img(test_path, target_size=(250, 250))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.array([test_image])
    predictions = model.predict(test_image) #get prediction of the test data
    plt.figure(figsize=(3, 3))
    plt.imshow(test_image.reshape(250, 250, 3) / 255.)
    plt.gray()
    print(predictions[0])
    plt.show()







