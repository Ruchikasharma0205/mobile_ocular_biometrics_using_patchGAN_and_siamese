import pandas as pd
import numpy as np
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

model = tf.keras.models.load_model(resource_path('my_model'))

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return img

def get_prediction(fname1, fname2):
    # return 1
    x = decode_img(tf.io.read_file(fname1))
    x1 = [x[:, : 256, :], x[:, 256 : 512, :], x[:, 512 : 768, :], x[:, 768 : 1024, :], x[:, 1024 : 1280, :]]

    x = decode_img(tf.io.read_file(fname2))
    x2 = [x[:, : 256, :], x[:, 256 : 512, :], x[:, 512 : 768, :], x[:, 768 : 1024, :], x[:, 1024 : 1280, :]]

    input_1_to_model = []
    input_2_to_model = []

    for i in range(5):
        for j in range(5):
            input_1_to_model.append(x1[i])
            input_2_to_model.append(x2[j])
    global model

    input_1_to_model = tf.convert_to_tensor(input_1_to_model, dtype=tf.float32)
    input_2_to_model = tf.convert_to_tensor(input_2_to_model, dtype=tf.float32)

    probs = model.predict([input_1_to_model, input_2_to_model], batch_size=25, use_multiprocessing=True)
    return np.mean(probs)

input_file_name = str(sys.argv[1])
# input_file_name = "test.csv"

input_files = np.array(pd.read_csv(input_file_name, header=None))

def get_testing_data():
  for i in range(input_files.shape[0]):
    fname1 = input_files[i][0]
    temp = fname1.split('/')
    temp = temp[len(temp) - 1]
    a,b,c = temp.split('_')
    p1_id = a
    x = decode_img(tf.io.read_file(fname1))
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x1 = [x[:, : 256, :], x[:, 256 : 512, :], x[:, 512 : 768, :], x[:, 768 : 1024, :], x[:, 1024 : 1280, :]]

    fname2 = input_files[i][1]
    temp = fname2.split('/')
    temp = temp[len(temp) - 1]
    a,b,c = temp.split('_')
    p2_id = a
    x = decode_img(tf.io.read_file(fname2))
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x2 = [x[:, : 256, :], x[:, 256 : 512, :], x[:, 512 : 768, :], x[:, 768 : 1024, :], x[:, 1024 : 1280, :]]

    yield {"input_1": x1[0], "input_2": x2[0]}

testing_dataset = tf.data.Dataset.from_generator(get_testing_data,
                                                 output_types=({"input_1": tf.float32, "input_2": tf.float32}),
                                                 output_shapes=({"input_1": (256, 256, 3), "input_2": (256, 256, 3)}))
testing_dataset = testing_dataset.batch(10)

result = model.predict(testing_dataset, verbose=1, use_multiprocessing=True)

result = pd.DataFrame(result)
result.to_csv('prediction.csv', header=False, index=False)
print('The generated match scores have been saved in prediction.csv')