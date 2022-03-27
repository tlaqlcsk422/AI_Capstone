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
from tensorflow_core.python.data.ops.dataset_ops import AUTOTUNE

img_path1 = "./시설 작물 질병 진단 이미지/Training/05.상추_0.정상"
img_path2 = "./시설 작물 질병 진단 이미지/Training/05.상추_1.질병"

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

Y=[]
for file in os.listdir(img_path2):
    file_path= os.path.join(img_path2,file)
    img = Image.open(file_path)
    img = img.resize((256,256),Image.ANTIALIAS)
    img = np.array(img)
    Y.append(img)

X = np.array(X)
Y= np.array(Y)

print(X.shape)
print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, random_state= 42, test_size = 0.25)

BUFFER_SIZE = 1000
BATCH_SIZE = 1

def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[256,256, 3])

  return cropped_image

# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image

def preprocess_image_train(image, label):
  image = random_jitter(image)
  image = normalize(image)
  return image

def preprocess_image_test(image, label):
  image = normalize(image)
  return image

X_train = X_train.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

y_train = y_train.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

X_test =X_test.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

y_test = y_test.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

sample_horse = next(iter(X_train))
sample_zebra = next(iter(y_train))

plt.subplot(121)
plt.title('Lettuce')
plt.imshow(sample_horse[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Lettuce')
plt.imshow(random_jitter(sample_horse[0]) * 0.5 + 0.5)

plt.subplot(121)
plt.title('Risk')
plt.imshow(sample_zebra[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Risk')
plt.imshow(random_jitter(sample_zebra[0]) * 0.5 + 0.5)

OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)



