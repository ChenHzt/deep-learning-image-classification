import glob
import os
from tkinter import *
from tkinter import filedialog

import PIL
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageTk, Image
from keras.applications.mobilenet import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

MODEL_FILE = 'model_cifar.model'
WIDTH = 224
HEIGHT = 224


def predict(model, img):
    """Run model prediction on image
    Args:
        model: keras model
        img: PIL format image
    Returns:
        list of predicted labels and their probabilities
    """
    new_image = image.load_img(img, target_size=(HEIGHT, WIDTH))
    x = image.img_to_array(new_image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    y_classes = preds.argmax(axis=-1)
    print(y_classes)
    folders_name = []
    folders = glob.glob("./dataset_GTSRB/Train/*")  # Reads all the folders in which images are present
    for i in folders:
        fn = os.path.basename(os.path.normpath(i))
        folders_name.append(fn)

    class_names = sorted(folders_name)  # Sorting them
    name_id_map = dict(zip(range(len(class_names)), class_names))

    return preds[0], name_id_map


def plot_preds(img, preds, cl_map):
    """Displays image and the top-n predicted probabilities in a bar graph
    Args:
        preds: list of predicted labels and their probabilities
    """
    # plt.rcdefaults()
    new_image = image.load_img(img, target_size=(HEIGHT, WIDTH))
    labels = []
    dict_labels_names = {
        0: "Speed limit (20km/h)",
        1: "Speed limit (30km/h)",
        2: "Speed limit (50km/h)",
        3: "Speed limit (60km/h)",
        4: "Speed limit (70km/h)",
        5: "Speed limit (80km/h)",
        6: "End of speed limit (80km/h)",
        7: "Speed limit (100km/h)",
        8: "Speed limit (120km/h)",
        9: "No passing",
        10: "No passing for vehicles over 3.5 metric tons",
        11: "Right-of-way at the next intersection",
        12: "Priority road",
        13: "Yield",
        14: "Stop",
        15: "No vehicles",
        16: "Vehicles over 3.5 metric tons prohibited",
        17: "No entry",
        18: "General caution",
        19: "Dangerous curve to the left",
        20: "Dangerous curve to the right",
        21: "Double curve",
        22: "Bumpy road",
        23: "Slippery road",
        24: "Road narrows on the right",
        25: "Road work",
        26: "Traffic signals",
        27: "Pedestrians",
        28: "Children crossing",
        29: "Bicycles crossing",
        30: "Beware of ice/snow",
        31: "Wild animals crossing",
        32: "End of all speed and passing limits",
        33: "Turn right ahead",
        34: "Turn left ahead",
        35: "Ahead only",
        36: "Go straight or right",
        37: "Go straight or left",
        38: "Keep right",
        39: "Keep left",
        40: "Roundabout mandatory",
        41: "End of no passing",
        42: "End of no passing by vehicles over 3.5 metric tons",
    }
    y_pos = np.arange(42 + 1)
    y = []
    for i in y_pos:
        y.append(cl_map[i])
        labels.append(dict_labels_names[i])
    y = list(map(int, y))
    # labels = (map(str, range(43 + 1)))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 8])
    plt.figure(figsize=(11, 15))
    plt.subplot(gs[0])
    plt.imshow(np.asarray(new_image))
    plt.subplot(gs[1])
    plt.barh(y, preds, align='center')
    plt.yticks(y_pos, labels)

    plt.xlabel('Probability')
    plt.title('Predictions')
    plt.tight_layout()
    path = './results/fig_{}.png'
    plt.savefig(path.format(os.path.basename(os.path.normpath(img))))
    plt.show()
    return path


def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(32, 32))
    img_tensor = image.img_to_array(img)  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor,
                                axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.  # imshow expects values in the range [0, 1]
    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


img_path = './images/t2.jpg'
model_path = './'
p_pred = './'


def file_browser_image():
    global img_path
    global p_pred
    print("here")
    img_path = filedialog.askopenfilename(initialdir="c://", title="Select file",
                                          filetypes=(
                                              ("jpeg files", "*.jpg"), ("png files", "*.png"), ("all files", "*.*")))
    model = load_model(model_path)

    preds, classes_map = predict(model, img_path)

    p_pred = plot_preds(img_path, preds, classes_map)


def file_browser_model():
    global model_path
    model_path = filedialog.askopenfilename(initialdir="c://", title="Select file")



root = Tk()
root.geometry("200x200")
root.resizable(0, 0)
frame = Frame(root)
frame.pack_propagate(0)
frame.pack(fill=BOTH, expand=1)
w1 = Label(frame, text="Hello,")
w1.pack()
button_model = Button(frame, text="choose a model file", command=lambda: file_browser_model())
button_model.pack()
button_image = Button(frame, text="choose an image file", command=lambda: file_browser_image())
button_image.pack()

root.mainloop()
