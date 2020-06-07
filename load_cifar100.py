import os

import imageio
import numpy as np
from keras.datasets import cifar100


(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
mean_image = np.mean(x_train, axis=0)
x_train=x_train-mean_image
mean_image = np.mean(x_test, axis=0)
x_test=x_test-mean_image


i=0
os.mkdir('./dataset_cifar100/train')
os.mkdir('./dataset_cifar100/test')
for y in range(100):
    os.mkdir('./dataset_cifar100/train/'+str(y))

indexes= np.zeros(100)

for img in x_train:
    imageio.imwrite('./dataset_cifar100/train/{}/{}.png'.format(y_train[i,0], "%05d" % (indexes[y_train[i,0]],)), img)
    indexes[y_train[i,0]]=indexes[y_train[i,0]]+1
    i = i+1

i=0
for y in range(100):
    os.mkdir('./dataset_cifar100/test/'+str(y))

indexes = np.zeros(100)

for img in x_test:
    imageio.imwrite('./dataset_cifar100/test/{}/{}.png'.format(y_test[i, 0], "%05d" % (indexes[y_test[i, 0]],)), img)
    indexes[y_test[i, 0]] = indexes[y_test[i, 0]] + 1
    i = i + 1
