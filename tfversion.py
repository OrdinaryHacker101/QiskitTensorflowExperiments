import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import cv2
import os

MALE = "C:\\Users\\Robin\\Desktop\\jasonBot\\male"
FEMALE = "C:\\Users\\Robin\\Desktop\\jasonBot\\female"
TRAIN_DATA = []

##for male in tqdm(os.listdir(MALE)):
##    try:
##        path = os.path.join(MALE, male)
##        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
##        img = cv2.resize(img, (64,64))
##        TRAIN_DATA.append([np.array(img)/255, np.array([0,1])])
####        plt.imshow(img)
####        plt.show()
##    except:
##        continue
##for female in tqdm(os.listdir(FEMALE)):
##    try:
##        path = os.path.join(FEMALE, female)
##        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
##        img = cv2.resize(img, (64,64))
##        TRAIN_DATA.append([np.array(img)/255, np.array([1,0])])
##    except:
##        continue
##
##np.save("train_data.npy", TRAIN_DATA)

TRAIN_DATA = np.load("train_data.npy", allow_pickle=True)
print(TRAIN_DATA)

np.random.shuffle(TRAIN_DATA)

print(len(TRAIN_DATA))

TRAIN = TRAIN_DATA[:-500]
TEST = TRAIN_DATA[-500:]

X = np.array([i[0] for i in TRAIN]).reshape(-1,64,64,1)
Y = np.array([i[1] for i in TRAIN])

test_x = np.array([i[0] for i in TEST]).reshape(-1,64,64,1)
test_y = np.array([i[1] for i in TEST])

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation="relu", input_shape=(64, 64, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(2, activation = "softmax"))

model.compile(optimizer="adam",
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

history = model.fit(X, Y, epochs=10,
                    validation_data = (test_x, test_y))

test_loss, test_acc = model.evaluate(test_x, test_y, verbose=10)

print("Final model accuracy: ", test_acc)
model.summary()

model.save_weights("C:\\Users\\Robin\\Desktop\\jasonBot\\checkpoints\\model_chkpt2")
