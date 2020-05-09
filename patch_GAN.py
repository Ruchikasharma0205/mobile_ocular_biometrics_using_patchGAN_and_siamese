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
Total images = 19516*5 of size 120*120*48 ====> 48 is the no. of filters
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
# all_images = all_images[: len(all_images) // 100]
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


clas = tf.keras.models.load_model('saved_model/left_right')


# In[ ]:


def flip_if_needed(x):
  imgs = [x[:, : 256, :], x[:, 256 : 512, :], x[:, 512 : 768, :], x[:, 768 : 1024, :], x[:, 1024 : 1280, :]]
  imgs = tf.convert_to_tensor(imgs, dtype=tf.float32)
  probs = clas.predict(imgs, batch_size=5, use_multiprocessing=True)
  prob = (probs[0] + probs[1] + probs[2] + probs[3] + probs[4]) / 5.0
  if prob < 0.5:
    x = tf.image.flip_left_right(x)
  return x

def get_lm_applied(img):
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.array([cv2.filter2D(x, -1, F[:,:,i]) for i in range(F.shape[2])])
    x = np.abs(x)
    img = np.zeros((256, 256, 144))
    for pos in range(48):
        img[:, :, 3 * pos] = x[pos, :, :, 0]
        img[:, :, 3 * pos + 1] = x[pos, :, :, 1]
        img[:, :, 3 * pos + 2] = x[pos, :, :, 2]
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    return img

# import time
# start = time.time()
# img = decode_img(tf.io.read_file(ids['1194'][0]))
# plt.imshow(img)
# plt.show()
# x = get_lm_applied(img[:, : 256, :])
# plt.imshow(x[:, :, 143])
# plt.show()
# img = flip_if_needed(img)
# plt.imshow(img)
# plt.show()
# x = get_lm_applied(img[:, : 256, :])
# plt.imshow(x[:, :, 143])
# plt.show()
# print(time.time() - start)


# In[ ]:


def get_training_data():
  global flag 
  while True:
#     flag += 1
#     if flag == 3:
#         flag = 0
    flag = 1 - flag
    if flag:
      p_id = train_ids_to_labels[random.randint(1, len(train_ids_to_labels)) - 1]
      pos = random.randint(1, len(ids[p_id])) - 1
      fname = ids[p_id][pos]
      temp = fname.split('/')
      temp = temp[len(temp) - 1]
      a,b,c = temp.split('_')
      e_id = b
      x = decode_img(tf.io.read_file(ids[p_id][pos]))
      if e_id == 'l':
        x = tf.image.flip_left_right(x)
      x = tf.convert_to_tensor(x, dtype=tf.float32)
      x1 = [x[:, : 256, :], x[:, 256 : 512, :], x[:, 512 : 768, :], x[:, 768 : 1024, :], x[:, 1024 : 1280, :]]
      pos = random.randint(1, len(ids[p_id])) - 1
      fname = ids[p_id][pos]
      temp = fname.split('/')
      temp = temp[len(temp) - 1]
      a,b,c = temp.split('_')
      e_id = b
      x = decode_img(tf.io.read_file(ids[p_id][pos]))
      if e_id == 'l':
        x = tf.image.flip_left_right(x)
      x = tf.convert_to_tensor(x, dtype=tf.float32)
      x2 = [x[:, : 256, :], x[:, 256 : 512, :], x[:, 512 : 768, :], x[:, 768 : 1024, :], x[:, 1024 : 1280, :]]
      for i in range(5):
        for j in range(5):
          yield {"input_1": x1[i], "input_2": x2[j]}, [1.0]
    else:
      fname1 = train_all_files[random.randint(1, len(train_all_files)) - 1]
      temp = fname1.split('/')
      temp = temp[len(temp) - 1]
      a,b,c = temp.split('_')
      p1_id = a
      e1_id = b
      x = decode_img(tf.io.read_file(fname1))
      if e1_id == 'l':
        x = tf.image.flip_left_right(x)
      x = tf.convert_to_tensor(x, dtype=tf.float32)
      x1 = [x[:, : 256, :], x[:, 256 : 512, :], x[:, 512 : 768, :], x[:, 768 : 1024, :], x[:, 1024 : 1280, :]]
      fname2 = train_all_files[random.randint(1, len(train_all_files)) - 1]
      temp = fname2.split('/')
      temp = temp[len(temp) - 1]
      a,b,c = temp.split('_')
      p2_id = a
      e2_id = b
      x = decode_img(tf.io.read_file(fname2))
      if e2_id == 'l':
        x = tf.image.flip_left_right(x)
      x = tf.convert_to_tensor(x, dtype=tf.float32)
      x2 = [x[:, : 256, :], x[:, 256 : 512, :], x[:, 512 : 768, :], x[:, 768 : 1024, :], x[:, 1024 : 1280, :]]
      # x2 = apply_MR_filter(x2)
      for i in range(5):
        for j in range(5):
          if p1_id == p2_id:
            yield {"input_1": x1[i], "input_2": x2[j]}, [1.0]
          else:
            yield {"input_1": x1[i], "input_2": x2[j]}, [0.0]


# In[ ]:


training_dataset = tf.data.Dataset.from_generator(get_training_data,
                                                 output_types=({"input_1": tf.float32, "input_2": tf.float32}, tf.float32),
                                                 output_shapes=(({"input_1": (256, 256, 3), "input_2": (256, 256, 3)}), 1))
training_dataset = training_dataset.batch(100)


# In[ ]:


# list(training_dataset.take(1).as_numpy_iterator())
# temp = list(training_dataset.take(4).as_numpy_iterator())


# In[ ]:


def get_validation_data():
  while True:
    fname1 = val_all_files[random.randint(1, len(val_all_files)) - 1]
    temp = fname1.split('/')
    temp = temp[len(temp) - 1]
    a,b,c = temp.split('_')
    p1_id = a
    x = decode_img(tf.io.read_file(fname1))
    x = flip_if_needed(x)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x1 = [x[:, : 256, :], x[:, 256 : 512, :], x[:, 512 : 768, :], x[:, 768 : 1024, :], x[:, 1024 : 1280, :]]

    fname2 = val_all_files[random.randint(1, len(val_all_files)) - 1]
    temp = fname2.split('/')
    temp = temp[len(temp) - 1]
    a,b,c = temp.split('_')
    p2_id = a
    x = decode_img(tf.io.read_file(fname2))
    x = flip_if_needed(x)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x2 = [x[:, : 256, :], x[:, 256 : 512, :], x[:, 512 : 768, :], x[:, 768 : 1024, :], x[:, 1024 : 1280, :]]

    for i in range(5):
      for j in range(5):
        if p1_id == p2_id:
          yield {"input_1": x1[i], "input_2": x2[j]}, [1.0]
        else:
          yield {"input_1": x1[i], "input_2": x2[j]}, [0.0]


# In[ ]:


validation_dataset = tf.data.Dataset.from_generator(get_validation_data,
                                                    output_types=({"input_1": tf.float32, "input_2": tf.float32}, tf.float32),
                                                    output_shapes=(({"input_1": (256, 256, 3), "input_2": (256, 256, 3)}), 1))
validation_dataset = validation_dataset.batch(100)


# In[ ]:


def get_testing_data():
  while True:
    fname1 = test_all_files[random.randint(1, len(test_all_files)) - 1]
    temp = fname1.split('/')
    temp = temp[len(temp) - 1]
    a,b,c = temp.split('_')
    p1_id = a
    x = decode_img(tf.io.read_file(fname1))
    x = flip_if_needed(x)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x1 = [x[:, : 256, :], x[:, 256 : 512, :], x[:, 512 : 768, :], x[:, 768 : 1024, :], x[:, 1024 : 1280, :]]

    fname2 = test_all_files[random.randint(1, len(test_all_files)) - 1]
    temp = fname2.split('/')
    temp = temp[len(temp) - 1]
    a,b,c = temp.split('_')
    p2_id = a
    x = decode_img(tf.io.read_file(fname2))
    x = flip_if_needed(x)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x2 = [x[:, : 256, :], x[:, 256 : 512, :], x[:, 512 : 768, :], x[:, 768 : 1024, :], x[:, 1024 : 1280, :]]

    for i in range(5):
      for j in range(5):
        if p1_id == p2_id:
          yield {"input_1": x1[i], "input_2": x2[j]}, [1.0]
        else:
          yield {"input_1": x1[i], "input_2": x2[j]}, [0.0]


# In[ ]:


testing_dataset = tf.data.Dataset.from_generator(get_testing_data,
                                                 output_types=({"input_1": tf.float32, "input_2": tf.float32}, tf.float32),
                                                 output_shapes=(({"input_1": (256, 256, 3), "input_2": (256, 256, 3)}), 1))

testing_dataset = testing_dataset.batch(100)


# In[ ]:


def downsample(filters, size, apply_batchnorm=False):

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.ReLU())

  return result


# In[ ]:


def Discriminator():

  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_1')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='input_2')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(32, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(64, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(128, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(256, 4, strides=1,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1)(zero_pad2) # (bs, 30, 30, 1)
  last = tf.keras.activations.sigmoid(last)
  last = tf.keras.layers.GlobalAveragePooling2D()(last)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)


# In[ ]:


disc = Discriminator()
disc.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', tf.keras.metrics.SensitivityAtSpecificity(0.99999),tf.keras.metrics.FalsePositives(),tf.keras.metrics.FalseNegatives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.TruePositives()])
disc.summary()


# In[ ]:


filepath = "patch_GAN_checkpoints" + "patch_GAN-{epoch:02d}-{val_sensitivity_at_specificity:.4f}.hdf5"
# filepath = "patch_GAN-{epoch:02d}-{val_sensitivity_at_specificity:.4f}.hdf5"
save_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_sensitivity_at_specificity', verbose=1, save_best_only=False,
    save_weights_only=False, mode='auto', save_freq='epoch')


# In[ ]:


disc.fit(x=training_dataset, epochs=20, validation_data=validation_dataset,
         steps_per_epoch=len(train_all_files) // 8,
         validation_steps=len(val_all_files) // 8,
         callbacks=[save_checkpoint], use_multiprocessing=True)


# In[ ]:


disc.save('saved_model/patch_GAN')


# In[ ]:


disc.evaluate(x=testing_dataset, verbose=1, steps=len(test_all_files) // 8, use_multiprocessing=True)

