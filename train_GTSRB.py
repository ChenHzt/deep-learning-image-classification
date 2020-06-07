import csv
import glob
import os
from random import shuffle
import keras
from keras.datasets import mnist
import numpy as np
from keras.applications import MobileNet
from keras.engine.saving import load_model
from keras.layers import Dense
from keras.models import Model
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
import shutil
from keras.callbacks import ModelCheckpoint


NEW_MODEL_FILE = 'model_gtsrb.model'

HEIGHT = 32
WIDTH = 32
images = []
labels = []
folders = glob.glob('./dataset_GTSRB/Train/*')

for folder in folders:
    fn = os.path.basename(os.path.normpath(folder))
    for f in glob.glob(folder + '/*.png'):
        fol = folder[:21] + '/' + fn
        images.append(fol + '\\' + os.path.basename(f))
        labels.append(fn)

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.50)

for i in range(len(x_test)):
    path= './dataset_GTSRB/Test/' + y_test[i]
    shutil.move(x_test[i], path)

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory('./dataset_GTSRB/Train',
                                                    target_size=(224, 224),
                                                    color_mode='rgb',
                                                    batch_size=20,
                                                    class_mode='categorical',
                                                    shuffle=True)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')

test_generator = train_datagen.flow_from_directory('./dataset_GTSRB/Test',
                                                   target_size=(224, 224),
                                                   color_mode='rgb',
                                                   batch_size=20,
                                                   class_mode='categorical',
                                                   shuffle=True)


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

        # This function is called at the end of each epoch

    def on_epoch_end(self, epoch, logs={}):
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            # plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label="train_loss")
            plt.plot(N, self.acc, label="train_acc")
            plt.plot(N, self.val_losses, label="val_loss")
            plt.plot(N, self.val_acc, label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig('./output/Epoch-{}.png'.format(epoch))
            plt.show()
            plt.close()



plot_losses = PlotLosses()


base_model = MobileNet(weights='imagenet',
                       include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)

x = Dense(512, activation='relu')(x)  # dense layer 3
preds = Dense(43, activation='softmax')(x)  # final layer with softmax activation
model = Model(inputs=base_model.input, outputs=preds)

for layer in model.layers[:]:
    layer.trainable = False

model.load_weights("./checkpoints/weights-improvement-03-0.91.hdf5")
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
step_size_train = train_generator.n // train_generator.batch_size
EPOCHS = 1
VALIDATION_STEPS = 40

# checkpoint
filepath = "./checkpoints/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint, plot_losses]

history = model.fit_generator(train_generator,
                              epochs=EPOCHS,
                              steps_per_epoch=step_size_train,
                              validation_data=test_generator,
                              validation_steps=VALIDATION_STEPS,
                              callbacks=callbacks_list)
model.save(NEW_MODEL_FILE)


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()


plot_training(history)
