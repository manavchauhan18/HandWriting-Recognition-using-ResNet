import matplotlib
matplotlib.use("Agg")

from tensorflow import keras
from tensorflow.keras import layers
from Models import ResNet
from keras_Dataset import load_Kaggle_Dataset
from keras_Dataset import load_mnist_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report
import numpy as np
import cv2

EPOCH = 15
INIT_LR = 1e-1
BS = 128

print("INFO[] Loading Data...")
(A_ZData, A_ZLabels) = load_Kaggle_Dataset("A_Z Handwritten Data.csv")
(DigitData, DigitLabels) = load_mnist_dataset()

A_ZLabels += 10

data = np.vstack([A_ZData, DigitData])
labels = np.hstack([A_ZLabels, DigitLabels])

data = [cv2.resize(image, (32,32)) for image in data]
data = np.array(data, dtype = "float32")

data = np.expand_dims(data, axis=-1)
data /= 255.0

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=None, random_state=42)

aug = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.05,
    shear_range=0.05,
    horizontal_flip=False,
    fill_mode="nearest"
)

print("INFO[] Compiling Model...")
opt = SGD(learning_rate=INIT_LR, decay= INIT_LR/EPOCH)

model = ResNet.build(32, 32, 1, len(lb.classes_), (3,3,3), (64, 64, 128, 256), reg=0.0005)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("INFO[] Training Model...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data = (testX, testY),
    steps_per_epoch  = len(trainX) // BS,
    epochs = EPOCH,
    class_weight = None,
    verbose = 1
)

labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

print("INFO[] Evaluating Model...")
predictions = model.predict(testX, batch_size = BS)
# print(classification_report(testY.argmax(axis=1),
# 	predictions.argmax(axis=1), target_names=labelNames))

print("[INFO] Serializing Network...")
model.save("OCR_detector.model", save_format="h5")

N = np.arange(0, EPOCH)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("Plot.png")