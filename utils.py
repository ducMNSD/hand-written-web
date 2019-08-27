import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
import os
from PIL import Image
import numpy as np
from keras.models import load_model
import cv2

BASE_DIR = os.path.dirname(__file__)
#UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/images')
MODEL_DIR = os.path.join(BASE_DIR, 'model/naruto.h5')

def load_keras_model():
	global model
	model = load_model(MODEL_DIR)
	# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
	# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
	# x_train /= 255
	# x_test /= 255
	# model.evaluate(x_test, y_test)

def load_images_to_data(image_directory):
    # list_of_files = os.listdir(image_directory)
    # for file in list_of_files:
    #     image_file_name = os.path.join(image_directory, file)
    #     if ".png" in image_file_name:
    img = Image.open(image_directory).convert("L")
    img = np.resize(img, (28,28,1))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,28,28,1)
          #  features_data = np.append(features_data, im2arr, axis=0)
           # label_data = np.append(label_data, [image_label], axis=0)
    return im2arr

def hienthi_kq(image_directory):
	img = load_images_to_data(image_directory)
	#cv2.imwrite('/home/duc_mnsd/Desktop/download.png',x_test[image_index])
	# plt.imshow(img.reshape(28, 28),cmap='Greys')
	pred = model.predict(img.reshape(1, 28,28, 1))
	kq = pred.argmax()
	return kq