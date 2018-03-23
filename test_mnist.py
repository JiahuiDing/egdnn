import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras import backend

# settings
batch_size = 256
num_classes = 10
epochs = 1000

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_train = x_train.astype('float32')
x_train /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)

x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_test /= 255
y_test = keras.utils.to_categorical(y_test, num_classes)


# model

'''
model = Sequential()
model.add(Dense(100, activation = 'relu', input_dim = 784))
model.add(Dense(num_classes, activation = 'softmax'))

sgd = keras.optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = keras.losses.categorical_crossentropy, optimizer = sgd, metrics = ['accuracy'])

model.fit(x_train, y_train,
			batch_size = batch_size,
			epochs = epochs,
			verbose = 2,
			validation_data = (x_test, y_test))
'''
