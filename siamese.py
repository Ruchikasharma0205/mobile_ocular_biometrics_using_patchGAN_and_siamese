#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[ ]:
import os
from glob import glob
all_images = glob('input/*/*/*/*.png')

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


# def get_validation_data():
#   global flag, clas
#   while True:
#     flag = 1 - flag
#     if flag:
#       p_id = val_ids_to_labels[random.randint(1, len(val_ids_to_labels)) - 1]
      
#       pos = random.randint(1, len(ids[p_id])) - 1
#       x = decode_img(tf.io.read_file(ids[p_id][pos]))
#       x = tf.convert_to_tensor(x, dtype=tf.float32)
#       x1 = [x[:, : 256, :], x[:, 256 : 512, :], x[:, 512 : 768, :], x[:, 768 : 1024, :], x[:, 1024 : 1280, :]]

#       pos = random.randint(1, len(ids[p_id])) - 1
#       x = decode_img(tf.io.read_file(ids[p_id][pos]))
#       x = flip_if_needed(x)
#       x = tf.convert_to_tensor(x, dtype=tf.float32)
#       x2 = [x[:, : 256, :], x[:, 256 : 512, :], x[:, 512 : 768, :], x[:, 768 : 1024, :], x[:, 1024 : 1280, :]]
      
#       for i in range(5):
#         for j in range(5):
#           yield {"input_1": x1[i], "input_2": x2[j]}, [1.0]
#     else:
#       fname1 = val_all_files[random.randint(1, len(val_all_files)) - 1]
#       temp = fname1.split('/')
#       temp = temp[len(temp) - 1]
#       a,b,c = temp.split('_')
#       p1_id = a
#       x = decode_img(tf.io.read_file(fname1))
#       x = flip_if_needed(x)
#       x = tf.convert_to_tensor(x, dtype=tf.float32)
#       x1 = [x[:, : 256, :], x[:, 256 : 512, :], x[:, 512 : 768, :], x[:, 768 : 1024, :], x[:, 1024 : 1280, :]]

#       fname2 = val_all_files[random.randint(1, len(val_all_files)) - 1]
#       temp = fname2.split('/')
#       temp = temp[len(temp) - 1]
#       a,b,c = temp.split('_')
#       p2_id = a
#       x = decode_img(tf.io.read_file(fname2))
#       x = flip_if_needed(x)
#       x = tf.convert_to_tensor(x, dtype=tf.float32)
#       x2 = [x[:, : 256, :], x[:, 256 : 512, :], x[:, 512 : 768, :], x[:, 768 : 1024, :], x[:, 1024 : 1280, :]]

#       for i in range(5):
#         for j in range(5):
#           if p1_id == p2_id:
#             yield {"input_1": x1[i], "input_2": x2[j]}, [1.0]
#           else:
#             yield {"input_1": x1[i], "input_2": x2[j]}, [0.0]


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


from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


# In[ ]:


def get_siamese_model(input_shape):
    """
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """
    initialize_weights = tf.random_normal_initializer(0., 1e-2)
    initialize_bias = tf.random_normal_initializer(0., 1e-2)
    # Define the tensors for the two input images
    left_input = tf.keras.layers.Input(input_shape, name='input_1')
    right_input = tf.keras.layers.Input(input_shape, name='input_2')
    
    # Convolutional Neural Network
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (10,10), strides=2, activation='relu', input_shape=input_shape,
                   kernel_initializer=initialize_weights, kernel_regularizer=l2(1e-3)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(64, (7,7), strides=2, activation='relu',
                     kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(1e-3)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(64, (4,4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(1e-3)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(128, (4,4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(1e-3)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2048, activation='sigmoid',
                   kernel_regularizer=l2(1e-2),
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias))
    
#     print(model.summary())
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = tf.keras.layers.Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = tf.keras.layers.Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = tf.keras.Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    return siamese_net


# In[ ]:


disc = get_siamese_model([256, 256, 3])
disc.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', tf.keras.metrics.SensitivityAtSpecificity(0.99999),tf.keras.metrics.FalsePositives(),tf.keras.metrics.FalseNegatives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.TruePositives()])
disc.summary()


# In[ ]:


filepath = "siamese_checkpoints/" + "siamese-{epoch:02d}-{val_sensitivity_at_specificity:.4f}.hdf5"
# filepath = "siamese_less_data_checkpoints/" + "siamese-{epoch:02d}-{val_sensitivity_at_specificity:.4f}.hdf5"
save_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_sensitivity_at_specificity', verbose=1, save_best_only=False,
    save_weights_only=False, mode='auto', save_freq='epoch')


# In[ ]:


disc.fit(x=training_dataset, epochs=20, validation_data=validation_dataset,
         steps_per_epoch=len(train_all_files) // 8,
         validation_steps=len(val_all_files) // 8,
         callbacks=[save_checkpoint], use_multiprocessing=True)


# In[ ]:


disc.save('saved_model/siamese')
# disc.save('saved_model/siamese_less_data')


# In[ ]:


disc.evaluate(x=testing_dataset, verbose=1, steps=len(test_all_files) // 8, use_multiprocessing=True)

