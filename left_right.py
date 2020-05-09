#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf


# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[ ]:


'''
Dataset description
Images of 150 persons from different angles
'''
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dataset_directory = 'input/'
dir_list = [None]*12
dir_list[0] = dataset_directory + 'v1_note4_dark/dark/'
dir_list[1] = dataset_directory + 'v1_note4_daylight/daylight/'
dir_list[2] = dataset_directory + 'v1_note4_office/office/'
dir_list[3] = dataset_directory + 'v1_oppo_dark/dark/'
dir_list[4] = dataset_directory + 'v1_oppo_daylight/daylight/'
dir_list[5] = dataset_directory + 'v1_oppo_office/office/'
dir_list[6] = dataset_directory + 'v2_note4_dark/dark/'
dir_list[7] = dataset_directory + 'v2_note4_daylight/daylight/'
dir_list[8] = dataset_directory + 'v2_note4_office/office/'
dir_list[9] = dataset_directory + 'v2_oppo_dark/dark/'
dir_list[10] = dataset_directory + 'v2_oppo_daylight/daylight/'
dir_list[11] = dataset_directory + 'v2_oppo_office/office/'
from glob import glob
all_images = []
for direc in dir_list:
    # print(len(os.listdir(direc + 'S1/')) + len(os.listdir(direc + 'S2/')))
    images = glob(direc+'S1/'+'*.png')
    all_images += images
    images = glob(direc+'S2/'+'*.png')
    all_images += images

#till here, we've got a list of names of all images
#image name: <id_of that_person>_<l/r>_<img_no.>.png


# In[ ]:


print(len(all_images))
# all_images = all_images[: len(all_images) // 10]
# print(len(all_images))


# In[ ]:


ids = {}
for img_name in all_images:
  # _,_,_,_,_,_,x = img_name.split('/')
  temp = img_name.split('/')
  # temp = temp[len(temp) - 1]
  # temp = temp.split('\\')
  x = temp[len(temp) - 1]
  a,b,c = x.split('_')
  if(a not in ids):
    ids[a]=[]
  ids[a].append(img_name)


# In[ ]:


# No of people
len(ids)


# In[ ]:


ids_to_labels = []
for key in ids:
  ids_to_labels.append(key)
  
import random
random.Random(42).shuffle(ids_to_labels)


# In[ ]:


val_size = 0.15
test_size = 0.15
test_ids_to_labels = ids_to_labels[: round(test_size * len(ids_to_labels))]
val_ids_to_labels = ids_to_labels[round(test_size * len(ids_to_labels)) : round(test_size * len(ids_to_labels)) + round(val_size * len(ids_to_labels))]
train_ids_to_labels = ids_to_labels[round(test_size * len(ids_to_labels)) + round(val_size * len(ids_to_labels)) : ]
print(len(train_ids_to_labels), len(val_ids_to_labels), len(test_ids_to_labels))


# In[ ]:


train_all_files = []
for id in train_ids_to_labels:
  for file_name in ids[id]:
    train_all_files.append(file_name)

val_all_files = []
for id in val_ids_to_labels:
  for file_name in ids[id]:
    val_all_files.append(file_name)

test_all_files = []
for id in test_ids_to_labels:
  for file_name in ids[id]:
    test_all_files.append(file_name)

random.Random(42).shuffle(train_all_files)
random.Random(42).shuffle(val_all_files)
random.Random(42).shuffle(test_all_files)
print(len(train_all_files) + len(val_all_files) + len(test_all_files))


# In[ ]:


BATCH_SIZE = 100


# In[ ]:


def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return img


# In[ ]:


flag = 0


# In[ ]:


def get_training_data():
  global flag
  while True:
    fname = train_all_files[random.randint(1, len(train_all_files)) - 1]
    temp = fname.split('/')
    temp = temp[len(temp) - 1]
    a,b,c = temp.split('_')
    eye_id = b
    x = decode_img(tf.io.read_file(fname))
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x1 = [x[:, : 256, :], x[:, 256 : 512, :], x[:, 512 : 768, :], x[:, 768 : 1024, :], x[:, 1024 : 1280, :]]

    for i in range(5):
      if eye_id == 'l':
        yield x1[i], [1.0]
      else:
        yield x1[i], [0.0]


# In[ ]:


training_dataset = tf.data.Dataset.from_generator(get_training_data,
                                                 output_types=(tf.float32, tf.float32),
                                                 output_shapes=((256, 256, 3), 1))
training_dataset = training_dataset.batch(100)


# In[ ]:


# list(training_dataset.take(1).as_numpy_iterator())
# temp = list(training_dataset.take(4).as_numpy_iterator())


# In[ ]:


def get_validation_data():
  global flag
  while True:
    fname = val_all_files[random.randint(1, len(val_all_files)) - 1]
    temp = fname.split('/')
    temp = temp[len(temp) - 1]
    a,b,c = temp.split('_')
    eye_id = b
    x = decode_img(tf.io.read_file(fname))
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x1 = [x[:, : 256, :], x[:, 256 : 512, :], x[:, 512 : 768, :], x[:, 768 : 1024, :], x[:, 1024 : 1280, :]]

    for i in range(5):
      if eye_id == 'l':
        yield x1[i], [1.0]
      else:
        yield x1[i], [0.0]


# In[ ]:


validation_dataset = tf.data.Dataset.from_generator(get_validation_data,
                                                    output_types=(tf.float32, tf.float32),
                                                    output_shapes=((256, 256, 3), 1))
validation_dataset = validation_dataset.batch(100)


# In[ ]:


def get_testing_data():
  global flag
  while True:
    fname = test_all_files[random.randint(1, len(test_all_files)) - 1]
    temp = fname.split('/')
    temp = temp[len(temp) - 1]
    a,b,c = temp.split('_')
    eye_id = b
    x = decode_img(tf.io.read_file(fname))
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x1 = [x[:, : 256, :], x[:, 256 : 512, :], x[:, 512 : 768, :], x[:, 768 : 1024, :], x[:, 1024 : 1280, :]]

    for i in range(5):
      if eye_id == 'l':
        yield x1[i], [1.0]
      else:
        yield x1[i], [0.0]


# In[ ]:


testing_dataset = tf.data.Dataset.from_generator(get_testing_data,
                                                 output_types=(tf.float32, tf.float32),
                                                 output_shapes=((256, 256, 3), 1))

testing_dataset = testing_dataset.batch(100)


# In[ ]:


def Classifier():

  inp = tf.keras.layers.Input(shape=[256, 256, 3])

  conv = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inp)
  conv = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv)
  conv = tf.keras.layers.Conv2D(32,(3,3),activation='relu')(conv)
  conv = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv)
  conv = tf.keras.layers.Conv2D(64,(3,3),activation='relu')(conv)
  conv = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv)
  conv = tf.keras.layers.Conv2D(64,(3,3),activation='relu')(conv)
  conv = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv)
  conv = tf.keras.layers.Conv2D(128,(3,3),activation='relu')(conv)
  conv = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv)
  conv = tf.keras.layers.Conv2D(128,(3,3),activation='relu')(conv)
  conv = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv)

  x = tf.keras.layers.Flatten()(conv)
  x = tf.keras.layers.Dense(64, activation='relu')(x)
  last = tf.keras.layers.Dense(1, activation='sigmoid')(x)

  return tf.keras.Model(inputs=inp, outputs=last)


# In[ ]:


clas = Classifier()
clas.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
clas.summary()


# In[ ]:


clas.fit(x=training_dataset, epochs=10, validation_data=validation_dataset,
         steps_per_epoch=len(train_all_files) // 40,
         validation_steps=len(val_all_files) // 40,
         callbacks=[], use_multiprocessing=True)


# In[ ]:


clas.save('saved_model/left_right')


# In[ ]:


testing_steps = len(test_all_files) // 40
clas.evaluate(x=testing_dataset, steps=testing_steps, use_multiprocessing=True)

