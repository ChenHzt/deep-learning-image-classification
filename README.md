# deep-learning-image-classification
semesterial project in the course "deep learning for computer vision"

Chosen classes for classification:
Bicycle, Bus, Motorcycle, Pickup truck, Traffic sign, Streetcar, Tractor, Train, Car.

Datasets in use:

- ImageNet: Real world image dataset which contains over million images
classified into 1000 classes.
The use of the ImageNet dataset was done by using the MobileNet model
which was originally trained on the ImageNet dataset.

- CIFAR-100: Consist of 100 classes, each class contains 600 32x32 colored
images (500 for training and 100 more for validating).
CIFAR-100 dataset was received from keras.datasets library as 2 tuples:
> x_train, x_test:
uint8 array of RGB image data with shape (num_samples, 3, 32, 32)

> y_train, y_test:
uint8 array of category labels (integers in range 0-9) with shape
(num_samples)
Those images were saved as .png, sorted in folders according to the
corresponded class and divided to train and validation images. From all of 100
classes of this dataset, 7 relevant classes were chosen: Bicycle, Bus,
Motorcycle, Pickup truck, Streetcar, Tractor and Train.
The Car class was taken from CIFAR-10 by using the same process as the
classes from CIFAR-100

- German Traffic Sign Recognition Benchmark (GTSRB): multi-class,
single-image classification challenge held at the International Joint Conference
on Neural Networks (IJCNN) 2011. Contains 43 classes of different traffic
signs and more than 50,000 32x32 colored images in total.the dataset was downloaded almost prepared for using. All the images were
sorted in folders according to the correct class, and the images were in .png
format.
Because this dataset is huge relatively to the number of images in class of the
CIFAR dataset, only 10-14 images of each class of the GTSRB were taken as
training data and also 100 random images were taken as validation data
