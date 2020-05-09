import numpy as np
import pandas as pd
import random

'''
Dataset description
Images of 150 persons from different angles
Total images = 19516*5 of size 120*120*48 ====> 48 is the no. of filters
'''
import os
dataset_directory = '/media/shubham/New Volume/DIP/VISOB2/input/'
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

ids_to_labels = []
for key in ids:
  ids_to_labels.append(key)
  
import random
random.Random(42).shuffle(ids_to_labels)

val_size = 0.15
test_size = 0.15
test_ids_to_labels = ids_to_labels[: round(test_size * len(ids_to_labels))]
val_ids_to_labels = ids_to_labels[round(test_size * len(ids_to_labels)) : round(test_size * len(ids_to_labels)) + round(val_size * len(ids_to_labels))]
train_ids_to_labels = ids_to_labels[round(test_size * len(ids_to_labels)) + round(val_size * len(ids_to_labels)) : ]
# print(len(train_ids_to_labels), len(val_ids_to_labels), len(test_ids_to_labels))

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
print(len(train_all_files), len(val_all_files), len(test_all_files))
print(len(train_all_files) + len(val_all_files) + len(test_all_files))

file_names = []
actual = []

for i in range(len(test_all_files)):
    fname1 = test_all_files[i]
    temp = fname1.split('/')
    temp = temp[len(temp) - 1]
    a,b,c = temp.split('_')
    p1_id = a
    for iter in range(2):
        j = random.randint(1, len(test_all_files)) - 1
        fname2 = test_all_files[j]
        temp = fname2.split('/')
        temp = temp[len(temp) - 1]
        a,b,c = temp.split('_')
        p2_id = a
        file_names.append([fname1, fname2])
        if p1_id == p2_id:
            actual.append(1)
        else:
            actual.append(0)

pd.DataFrame(file_names).to_csv('test.csv', header=False, index=False)
pd.DataFrame(actual).to_csv('actual.csv', header=False, index=False)