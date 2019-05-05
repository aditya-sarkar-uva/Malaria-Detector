from __future__ import print_function
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os

original_parasitized = 'C:\\Users\\reach\\Documents\\Stuff\\Machine Learning\\Malaria-Detector\\original_cell_images\\Parasitized\\'
original_uninfected = 'C:\\Users\\reach\\Documents\\Stuff\\Machine Learning\\Malaria-Detector\\original_cell_images\\Uninfected\\'
resized_parasitized = 'C:\\Users\\reach\\Documents\\Stuff\\Machine Learning\\Malaria-Detector\\resized_cell_images\\Parasitized\\'
resized_uninfected = 'C:\\Users\\reach\\Documents\\Stuff\\Machine Learning\\Malaria-Detector\\resized_cell_images\\Uninfected\\'
batch_size, num_classes, epochs = 128, 2, 12
img_rows, img_cols = 64, 64

def resize_images():
	image_count = 0
	for filename in os.listdir(original_parasitized):
		if filename != "Thumbs.db":
			img = load_img(original_parasitized + filename, target_size=(img_rows, img_cols))
			img_array = img_to_array(img)
			save_img(resized_parasitized + filename, img_array)
			image_count += 1
			if image_count % 1000 == 0:
				print("Resizing parasitized image " + str(image_count))

	image_count = 0
	for filename in os.listdir(original_uninfected):
		if filename != "Thumbs.db":
			img = load_img(original_uninfected + filename, target_size=(img_rows, img_cols))
			img_array = img_to_array(img)
			save_img(resized_uninfected + filename, img_array)
			image_count += 1
			if image_count % 1000 == 0:
				print("Resizing uninfected image " + str(image_count))

resize_images()
