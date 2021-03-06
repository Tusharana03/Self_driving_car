# -*- coding: utf-8 -*-
"""Behavioural_cloning

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lVAkgxHjudyLM9qBWxvr9Qk4rb9zts7G
"""

# Importing the data generated from UDACITY uploded to github 

!git clone https://github.com/rslim087a/track.git

!ls track

!pip3 install imgaug

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
import cv2
import pandas as pd
import random
import os
import ntpath
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
from imgaug import augmenters as iaa

# Reading the data

datadir = "track"
columns = ['center', 'left', 'right','steering', 'throttle', 'reverse','speed' ]
data = pd.read_csv(os.path.join(datadir,"driving_log.csv"), names=columns)
pd.set_option('display.max_colwidth',-1)
data.head()

# Reducing the path labels of the images

def path_leaf(path):
  head, tail = ntpath.split(path)
  return tail

data['center'] = data['center'].apply(path_leaf)
data['left'] = data["left"].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)
data.head()

# plotting histogram for the steering values

num_of_bins = 25
samples_per_bins = 450
hist, bins = np.histogram(data['steering'], num_of_bins)
center = (bins[:-1] + bins[1:])*0.5
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']),np.max(data['steering'])),(samples_per_bins, samples_per_bins))

# reducing the data which corresponds to zero steering angle

print('total data', len(data))
remove_list = []
for j in range(num_of_bins):
  list_ = []
  for i in range(len(data['steering'])):
    if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
      list_.append(i)
  list_ = shuffle(list_)
  list_ = list_[samples_per_bins:]
  remove_list.extend(list_)

print('removed', len(remove_list))
data.drop(data.index[remove_list],inplace = True)
print('remaining', len(data))
hist , _ = np.histogram(data['steering'], num_of_bins)
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']),np.max(data['steering'])),(samples_per_bins, samples_per_bins))

# taking the images only from the center camera

def load_img_steering(datadir,df):
  image_path = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
    image_path.append(os.path.join(datadir, center.strip()))
    steering.append(float(indexed_data[3]))
  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths , steerings


image_paths , steerings = load_img_steering(datadir + '/IMG', data)

# Splitting the data into training and validation data set

x_train, x_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size= 0.2, random_state=0,)
print('training samples %s , valid samples %s'%(len(x_train),len(x_valid)))

fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
axes[0].hist(y_train, bins= num_of_bins, width= 0.05, color= 'blue')
axes[0].set_title('Training set')
axes[1].hist(y_valid, bins= num_of_bins, width = 0.05, color = 'red')
axes[1].set_title('Validation set')

# creating functions to generate new data by modifying images

def zoom(image):
  zoom = iaa.Affine(scale=(1,1.3))
  image = zoom.augment_image(image)
  return image

image = image_paths[random.randint(0,1000)]
original_image = mpimg.imread(image)
zoomed_image = zoom(original_image)
fig,axes = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout
axes[0].imshow(original_image)
axes[0].set_title('original')
axes[1].imshow(zoomed_image)
axes[1].set_title('zoomed')

def img_random_brightness(image):
  brightness = iaa.Multiply((0.2,1.2))
  image = brightness.augment_image(image)
  return image

image = image_paths[random.randint(0,1000)]
original_image = mpimg.imread(image)
panned_image = img_random_brightness(original_image)
fig,axes = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout
axes[0].imshow(original_image)
axes[0].set_title('original')
axes[1].imshow(panned_image)
axes[1].set_title('zoomed')

def img_random_flip(image, steering_angle):
  image = cv2.flip(image, 1)
  steering_angle = -steering_angle
  return image, steering_angle


random_index = random.randint(0,1000)
image = image_paths[random_index]
steering_angle = steerings[random_index]

original_image = mpimg.imread(image)
flipped_image, flipped_steering_angle = img_random_flip(original_image, steering_angle)
fig,axes = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout
axes[0].imshow(original_image)
axes[0].set_title('original' + 'steering' + str(steering_angle))
axes[1].imshow(flipped_image)
axes[1].set_title('flipped'+ 'steering' + str(flipped_steering_angle))

def pan(image):
  pan = iaa.Affine(translate_percent={'x':(-0.1,0.1) ,'y':(-0.1,0.1)})
  image = pan.augment_image(image)
  return image

image = image_paths[random.randint(0,1000)]
original_image = mpimg.imread(image)
zoomed_image = pan(original_image)
fig,axes = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout
axes[0].imshow(original_image)
axes[0].set_title('original')
axes[1].imshow(zoomed_image)
axes[1].set_title('panned')

# combining all the modifying functions into one function

def random_augment(image, steering_angle):
  image = mpimg.imread(image)

  if np.random.rand() < 0.5:
    image = pan(image)
  if np.random.rand() < 0.5:
    image = zoom(image)
  if np.random.rand() < 0.5:
    image = img_random_brightness(image)
  if np.random.rand() < 0.5:
    image, steering_angle = img_random_flip(image, steering_angle)

  return image, steering_angle


# visualizing modified data

ncol = 2
nrow = 10

fig, axs = plt.subplots(nrows=nrow,ncols=ncol, figsize=(15,50))
fig.tight_layout
for i in range(10):
  randnum = random.randint(0, len(image_paths)-1)
  random_image = image_paths[randnum]
  random_steering = steerings[randnum]

  original_image = mpimg.imread(random_image)
  augmented_image, steering = random_augment(random_image, random_steering)

  axs[i][0].imshow(original_image)
  axs[i][0].set_title('original')
  axs[i][1].imshow(augmented_image)
  axs[i][1].set_title('augmented')

# preprocessing the images

def img_preprocessing(img):
  #img = mpimg.imread(img)
  img = img[60:135,:,:]
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  img = cv2.GaussianBlur(img, (3,3), 0)
  img = cv2.resize(img, (200,66))
  img = img/255


  return img



"""image = image_paths[100]
original_image = mpimg.imread(image)
preprocessed_image = img_preprocessing(image)
fig , axes = plt.subplots(nrows=1, ncols=2, figsize= (15,10))
fig.tight_layout()
axes[0].imshow(original_image)
axes[0].set_title('original')
axes[1].imshow(preprocessed_image)
axes[1].set_title('preprocessed image')"""

# defining the function to generate the data

def batch_generator(image_paths,steering_ang,batch_size, istraining):
  while True:
    batch_img = []
    batch_steering = []
    for i in range(batch_size):
      random_index = random.randint(0, len(image_paths)-1)

      if istraining:
        im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
      else:
        im = mpimg.imread(image_paths[random_index])
        steering = steering_ang[random_index]

      im = img_preprocessing(im)
      batch_img.append(im)
      batch_steering.append(steering)
    yield (np.asarray(batch_img), np.asarray(batch_steering))

# applying the data generator to check

x_train_gen, y_train_gen = next(batch_generator(x_train,y_train,1,1))
x_val_gen, y_val_gen = next(batch_generator(x_valid,y_valid,1,0))

fig,axes = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()
axes[0].imshow(x_train_gen[0])
axes[0].set_title('training')
axes[1].imshow(x_val_gen[0])
axes[1].set_title('valid')


plt.imshow(x_train[random.randint(0,len(x_train)-1)])
plt.axis('off')
print(x_train.shape)


# Creating the Convolutional network 

def nvidia_model():
  model = Sequential()
  model.add(Convolution2D(filters=24,kernel_size=(5,5),strides=(2,2),input_shape=(66,200,3), activation='elu'))
  model.add(Convolution2D(filters=36, kernel_size=(5,5),strides=(2,2),activation='elu'))
  model.add(Convolution2D(filters=48, kernel_size=(5,5),strides=(2,2),activation='elu'))
  model.add(Convolution2D(filters=64, kernel_size=(3,3),activation='elu'))
  model.add(Convolution2D(filters=64, kernel_size=(3,3),activation='elu'))
  #model.add(Dropout(0.5))

  model.add(Flatten())

  model.add(Dense(units=100, activation='elu'))
  model.add(Dropout(0.5))
  model.add(Dense(units=50, activation='elu'))
  #model.add(Dropout(0.5))
  model.add(Dense(units=10, activation='elu'))
  #model.add(Dropout(0.5))
  model.add(Dense(units=1))
  model.compile(Adam(learning_rate=0.001),loss='mse')
  return model

model = nvidia_model()
print(model.summary())

# training the model while generating new data

history = model.fit_generator(batch_generator(x_train, y_train, 100,1),steps_per_epoch=345,epochs=10,validation_data=batch_generator(x_valid,y_valid,100,0), validation_steps=200,verbose=1,shuffle=1)

#history = model.fit(x_train,y_train,epochs=30, validation_data=(x_valid,y_valid), batch_size=100,verbose=1, shuffle=1)

# plotting the losses

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['traning', 'validation'])

# saving and downloading the model 

model.save('model6.h5')

from google.colab import files
files.download('model6.h5')