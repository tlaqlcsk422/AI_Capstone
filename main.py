import tensorflow as tf
import json
import os
import cv2
from PIL import Image
import numpy as np
import sys
import sklearn as sk
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Flatten,MaxPooling2D,Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


print(device_lib.list_local_devices())

# gpu 확인하기
print(sys.version)
print(pd.__version__)
print(sk.__version__)
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is","available" if gpu else "NOT AVAILABLE")

img_path1 = "./시설 작물 질병 진단 이미지/Training/05.상추_0.정상"
img_path2 = "./시설 작물 질병 진단 이미지/Training/05.상추_1.질병"

label_path1 = "./시설 작물 질병 진단 이미지/Training/[라벨]05.상추_0.정상"
label_path2 = "./시설 작물 질병 진단 이미지/Training/[라벨]05.상추_1.질병"

#이미지 받아오기
size = 256,256
X = []
for file in os.listdir(img_path1):
    file_path= os.path.join(img_path1,file)
    img = Image.open(file_path)
    img = img.resize((256,256),Image.ANTIALIAS)
    img = np.array(img)
    X.append(img)
    #img.show() 이미지 확인


for file in os.listdir(img_path2):
    file_path= os.path.join(img_path2,file)
    img = Image.open(file_path)
    img = img.resize((256,256),Image.ANTIALIAS)
    img = np.array(img)
    X.append(img)

X = np.array(X)
print(X.shape)


#라벨값 받아오기
Y = []
for file in os.listdir(label_path1):
    dir_path = label_path1 + "/" + file
    with open(dir_path, "r", encoding="utf8") as f:
        contents = f.read() # string 타입
        json_data = json.loads(contents)
    Y.append(json_data["annotations"]["risk"])

for file in os.listdir(label_path2):
    dir_path = label_path2 + "/"  + file
    with open(dir_path, "r", encoding="utf8") as f:
        contents = f.read() # string 타입
        json_data = json.loads(contents)
    Y.append(json_data["annotations"]["risk"])

Y = np.array(Y)
# plt.hist(Y, bins=4, label='Lettuce Risk')
# plt.show()

Y = tf.keras.utils.to_categorical(Y)
print(Y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y,
                                                    random_state= 42, test_size = 0.25)


X_train = X_train/255.0
X_test = X_test/255.0

model = Sequential([
    Conv2D(32,(3,3),strides = 2,input_shape = (256 ,256, 3), activation = 'relu'),
    MaxPooling2D(),
    Conv2D(32, (3,3), activation = 'relu'),
    MaxPooling2D(),
    Conv2D(32, (3,3), activation = 'relu'),
    MaxPooling2D(),
    Conv2D(32, (3,3), activation = 'relu'),
    MaxPooling2D(),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation = 'relu'),
    Dropout(0.5),
    Dense(64, activation = 'relu'),
    Dropout(0.5),
    Dense(4, activation = 'softmax')
])
model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs = 5, validation_data=(X_test, y_test))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc=0)

plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc=0)

plt.show()
