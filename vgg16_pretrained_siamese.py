#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf


# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[ ]:


import os
from glob import glob
all_images = glob('input/*/*/*/*.png')


# In[ ]:


print(len(all_images))


# In[ ]:


ids = {}
for img_name in all_images:
  temp = img_name.split('/')
  x = temp[len(temp) - 1]
  a,b,c = x.split('_')
  if(a not in ids):
    ids[a]=[]
  ids[a].append(img_name)


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


def decode_img(img):
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
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


def get_validation_data():
  while True:
    fname1 = val_all_files[random.randint(1, len(val_all_files)) - 1]
    temp = fname1.split('/')
    temp = temp[len(temp) - 1]
    a,b,c = temp.split('_')
    p1_id = a
    x = decode_img(tf.io.read_file(fname1))
    # x = flip_if_needed(x)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x1 = [x[:, : 256, :], x[:, 256 : 512, :], x[:, 512 : 768, :], x[:, 768 : 1024, :], x[:, 1024 : 1280, :]]
    # x1 = apply_MR_filter(x1)
    fname2 = val_all_files[random.randint(1, len(val_all_files)) - 1]
    temp = fname2.split('/')
    temp = temp[len(temp) - 1]
    a,b,c = temp.split('_')
    p2_id = a
    x = decode_img(tf.io.read_file(fname2))
    # x = flip_if_needed(x)
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
    # x = flip_if_needed(x)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x1 = [x[:, : 256, :], x[:, 256 : 512, :], x[:, 512 : 768, :], x[:, 768 : 1024, :], x[:, 1024 : 1280, :]]
    # x1 = apply_MR_filter(x1)
    fname2 = test_all_files[random.randint(1, len(test_all_files)) - 1]
    temp = fname2.split('/')
    temp = temp[len(temp) - 1]
    a,b,c = temp.split('_')
    p2_id = a
    x = decode_img(tf.io.read_file(fname2))
    # x = flip_if_needed(x)
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


testing_dataset = tf.data.Dataset.from_generator(get_testing_data,
                                                 output_types=({"input_1": tf.float32, "input_2": tf.float32}, tf.float32),
                                                 output_shapes=(({"input_1": (256, 256, 3), "input_2": (256, 256, 3)}), 1))

testing_dataset = testing_dataset.batch(100)


# In[ ]:


from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


# In[ ]:



from tensorflow.keras.applications import VGG16
model = VGG16(input_shape=[256,256,3],weights='imagenet', include_top=False)
for layer in model.layers:
    layer.trainable=False
model.summary()


# In[ ]:


def get_siamese_model(input_shape):
    initialize_weights = tf.random_normal_initializer(0., 1e-2)
    initialize_bias = tf.random_normal_initializer(0., 1e-2)
    # Define the tensors for the two input images
    left_input = tf.keras.layers.Input([256,256,3], name='input_1')
    right_input = tf.keras.layers.Input([256,256,3], name='input_2')

    # Convolutional Neural Network
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(model.output)
    x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(2048, activation='sigmoid',
    #                 kernel_regularizer=l2(1e-3),
    #                 kernel_initializer=initialize_weights,bias_initializer=initialize_bias)(x)
    # x = tf.keras.layers.Dense(1024,activation='sigmoid',
    #                 kernel_regularizer=l2(1e-3),
    #                 kernel_initializer=initialize_weights,bias_initializer=initialize_bias)(x)
    # x = tf.keras.layers.Dense(512,activation='sigmoid',
    #                 kernel_regularizer=l2(1e-3),
    #                 kernel_initializer=initialize_weights,bias_initializer=initialize_bias)(x)
    x = tf.keras.layers.Dense(256,activation='sigmoid',
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer=initialize_weights,bias_initializer=initialize_bias)(x)
    # print(model1.summary())

    # Generate the encodings (feature vectors) for the two images
    final_model=tf.keras.Model(inputs=model.input,outputs=x)
    final_model.summary()
    encoded_l = final_model(left_input)
    encoded_r = final_model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = tf.keras.layers.Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = tf.keras.layers.Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)
    # prediction = tf.keras.layers.GlobalAveragePooling2D()(L1_distance)
    # Connect the inputs with the outputs
    siamese_net = tf.keras.Model(inputs=[left_input,right_input],outputs=prediction)
      
      # return the model
    return siamese_net


# In[ ]:


disc = get_siamese_model([256, 256, 48])
disc.compile(optimizer=tf.keras.optimizers.Adam(lr=.001), loss='binary_crossentropy', metrics=['binary_accuracy', tf.keras.metrics.SensitivityAtSpecificity(0.99999),tf.keras.metrics.FalsePositives(),tf.keras.metrics.FalseNegatives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.TruePositives()])
disc.summary()


# In[ ]:


filepath = "siamese_checkpoints/" + "siamese-{epoch:02d}-{val_sensitivity_at_specificity:.4f}.hdf5"
# filepath = "siamese_less_data_checkpoints/" + "siamese-{epoch:02d}-{val_sensitivity_at_specificity:.4f}.hdf5"
save_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_sensitivity_at_specificity', verbose=1, save_best_only=False,
    save_weights_only=False, mode='auto', save_freq='epoch')


# In[ ]:


disc.fit(x=training_dataset, epochs=20,validation_data=validation_dataset,
         steps_per_epoch=len(train_all_files) // 8,
         validation_steps=len(val_all_files) // 8,
         callbacks=[save_checkpoint], use_multiprocessing=True)


# In[ ]:


disc.save('saved_model/vgg16_pretrained_siamese')


# In[ ]:


disc.evaluate(x=testing_dataset, verbose=1, steps=len(test_all_files) // 8, use_multiprocessing=True)

