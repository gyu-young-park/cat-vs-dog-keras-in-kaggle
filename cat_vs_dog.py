# -*- coding: utf-8 -*-
"""
Created on Wed May 29 20:57:48 2019

@author: rbdud
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os
from random import shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm

trainDir ='train//'
testDir ='test1//'
CHANNELS = 3
batchSize=16
imgSize = 150
color=True


def pre_data(images):
    count = len(images)
    #count = len(images)
    X = np.ndarray((count, imgSize, imgSize, CHANNELS), dtype=np.float32)
    y = np.zeros((count,), dtype=np.float32)
    
    for i, image_file in enumerate(images):
        img = image.load_img(image_file, target_size=(imgSize, imgSize))
        X[i] = image.img_to_array(img)
        if 'dog' in image_file:
            y[i] = 1.
        if i%1000 == 0: print('Processed {} of {}'.format(i, count))
    
    return X, y

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(imgSize, imgSize, CHANNELS)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
#     model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Conv2D(256, (3, 3), activation='relu'))
#     model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dense(1000, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

if __name__ == '__main__':
    # 만약 이미 있다면
    # X_train = np.load('train_xdata.npy')
    # y_train = np.load('train_ydata.npy')
    # X_validation = np.load('validation_xdata.npy')
    # y_validation = np.load('validation_ydata.npy')
    # 이미 데이터가 있다면, pre_data도 하지 않고 dataset을 읽지도 않는다.
    #get dataset
    original_train_images = [trainDir+i for i in os.listdir(trainDir)]
    train_dogs =   [trainDir+i for i in os.listdir(trainDir) if 'dog' in i]
    train_cats =   [trainDir+i for i in os.listdir(trainDir) if 'cat' in i]
    test_images =  [testDir+i for i in os.listdir(testDir)]
    # cut off the dataset
    original_train_images = train_dogs[:10] + train_cats[:10]
    shuffle(original_train_images)
    test_images =  test_images[:10]

    train_images = original_train_images[:10]
    validation_images = original_train_images[10:]

    X_train, y_train = pre_data(train_images)
    X_validation, y_validation = pre_data(validation_images)

    #augmentation
    train_datagen = image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,)

    validation_datagen = image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow(
        X_train,
        y_train,
        batch_size=batchSize)

    validation_generator = validation_datagen.flow(
        X_validation,
        y_validation,
        batch_size=batchSize)

    #np.save('train_xdata.npy',X_train)
    #np.save('train_ydata.npy',y_train)
    #np.save('validation_xdata.npy',X_validation)
    #np.save('validation_ydata.npy',y_validation)

    plt.figure(figsize=(30, 30))
    for i in range(9):
        x, y = train_generator.next()
        plt.subplot(2, 10, i+1)
        plt.imshow(x[i])
        #plot_data(X_train[i:], y_train[i:], 6)
    plt.show()



    model = create_model()
    model.summary()

    model.compile(loss='binary_crossentropy',
                 optimizer=optimizers.RMSprop(lr=1e-4),
                 metrics=['accuracy'])

    train_steps = len(train_images)/batchSize
    validation_steps = len(validation_images)/batchSize

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        verbose=2)

    model.save('dogs-v-cat-data-augmentation-04.h5')
    evaluation_images = train_dogs[1000:20] + train_cats[1000:20]
    shuffle(evaluation_images)

    X_evaluation, y_evaluation = pre_data(evaluation_images)
    X_evaluation /= 255
    evaluation = model.evaluate(X_evaluation, y_evaluation)
    print(evaluation)
    X_test, _ = pre_data(test_images)
    X_test /= 255
    predictions = model.predict(X_test)
    for i in range(0,10):
        if predictions[i, 0] >= 0.5:
            print('I am {:.2%} sure this is a Dog'.format(predictions[i][0]))
        else:
            print('I am {:.2%} sure this is a Cat'.format(1-predictions[i][0]))

        plt.imshow(image.array_to_img(X_test[i]))
        plt.show()