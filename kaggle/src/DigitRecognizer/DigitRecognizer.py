# -*- coding:UTF-8 -*-
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
from keras import regularizers
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


batch_size = 86
num_classes = 10 
epochs = 30
seed = 7

train = pd.read_csv('train.csv').values
X = train[:, 1:].astype('float32') 
X /= 255
Y = np_utils.to_categorical(train[:,0].astype('int32'), num_classes) # labels

trainX, validationX, trainY, validationY = train_test_split(X, Y, test_size=0.1, random_state=seed)

rows, cols = 28, 28
trainX = trainX.reshape(trainX.shape[0], rows, cols, 1)
validationX = validationX.reshape(validationX.shape[0], rows, cols, 1)
input_shape = (rows, cols, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), activation='relu', padding = 'Same', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(5,5), activation='relu', padding = 'Same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding = 'Same', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding = 'Same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                             samplewise_center=False,  # set each sample mean to 0
                             featurewise_std_normalization=False,  # divide inputs by std of the dataset
                             samplewise_std_normalization=False,  # divide each input by its std
                             zca_whitening=False,  # apply ZCA whitening
                             rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                             zoom_range = 0.1, # Randomly zoom image 
                             width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                             height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                             horizontal_flip=False,  # randomly flip images
                             vertical_flip=False)  # randomly flip images

datagen.fit(trainX)

history = model.fit_generator(datagen.flow(trainX, trainY, batch_size=batch_size),
                              epochs = epochs, 
                              validation_data = (validationX, validationY),
                              verbose = 2, 
                              steps_per_epoch=trainX.shape[0] // batch_size,
                              callbacks=[learning_rate_reduction])

#model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=batch_size, epochs=epochs, verbose=2)

score = model.evaluate(trainX, trainY,verbose=2)
print('Train accuracy:', score[1])

testX = pd.read_csv('test.csv').values.astype('float32')
testX /= 255
testX = testX.reshape(testX.shape[0],rows,cols, 1)

testY = model.predict_classes(testX, verbose=2)
pd.DataFrame({"ImageId": list(range(1,len(testY)+1)),"Label": testY}).to_csv('output.csv', index=False, header=True)