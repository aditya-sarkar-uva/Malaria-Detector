from __future__ import print_function
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split
import os

original_parasitized = 'C:\\Users\\reach\\Documents\\Stuff\\Machine Learning\\Malaria-Detector\\original_cell_images\\Parasitized\\'
original_uninfected = 'C:\\Users\\reach\\Documents\\Stuff\\Machine Learning\\Malaria-Detector\\original_cell_images\\Uninfected\\'
resized_parasitized = 'C:\\Users\\reach\\Documents\\Stuff\\Machine Learning\\Malaria-Detector\\resized_cell_images\\Parasitized\\'
resized_uninfected = 'C:\\Users\\reach\\Documents\\Stuff\\Machine Learning\\Malaria-Detector\\resized_cell_images\\Uninfected\\'
batch_size, num_classes, epochs = 128, 2, 1
channels, img_rows, img_cols = 3, 32, 32

def resize_images():
	image_count = 0
	for filename in os.listdir(original_parasitized):
		if filename != "Thumbs.db":
			img = load_img(original_parasitized + filename, target_size=(img_rows, img_cols))
			img_array = img_to_array(img)
			save_img(resized_parasitized + filename, img_array)
			image_count += 1
			if image_count % 1000 == 0:
				print("Resizing parasitized cell image " + str(image_count))

	print("Finished resizing parasitized cell images")

	image_count = 0
	for filename in os.listdir(original_uninfected):
		if filename != "Thumbs.db":
			img = load_img(original_uninfected + filename, target_size=(img_rows, img_cols))
			img_array = img_to_array(img)
			save_img(resized_uninfected + filename, img_array)
			image_count += 1
			if image_count % 1000 == 0:
				print("Resizing uninfected image " + str(image_count))

	print("Finished resizing uninfected cell images")
	
def create_datasets():
	data = np.ndarray(shape=(27558, img_rows, img_cols, channels), dtype=np.float32)
	image_count = 0
	for filename in os.listdir(resized_parasitized):
		img = load_img(resized_parasitized + filename)
		img_array = img_to_array(img) # shape is (64, 64, 3)
		img_array.reshape((img_rows, img_cols, channels))
		data[image_count] = img_array
		image_count += 1
	
	print("Finished gathering parasitized cell images")

	image_count = 0
	for filename in os.listdir(resized_uninfected):
		img = load_img(resized_uninfected + filename)
		img_array = img_to_array(img) # shape is (64, 64, 3)
		img_array.reshape((img_rows, img_cols, channels))
		data[image_count] = img_array
		image_count += 1
	
	print("Finished gathering uninfected cell images")

	input_shape = (img_rows, img_cols, 3)

	labels = np.ndarray(shape=(27558), dtype=int)
	for n in range(0, 13779):
		labels[n] = 1
	for n in range(13779, 27558):
		labels[n] = 0
	
	print("Generated labels")

	idx = np.random.permutation(len(data))
	print(idx)
	data, labels = data[idx], labels[idx]

	print("Shuffled data and labels")

	data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.20)
	
	print("Split training and test sets")

	data_train /= 255
	data_test /= 255

	print("Normalized data")

	labels_train = keras.utils.to_categorical(labels_train, num_classes)
	labels_test = keras.utils.to_categorical(labels_test, num_classes)

	print("One-hot encoded labels")

	return data_train, data_test, labels_train, labels_test, input_shape

def train_model(input_shape):
	model = Sequential()
	#first argument is number of convolution filters, next is row and column size of kernels
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	# reduces number of parameters by sliding a 2x2 pooling filter across previous layer and taking the max of the 4 values in the filter
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# prevents overfitting
	model.add(Dropout(0.25))
	# weights from the convolution layers are flattened into 1 dimension before passing to the fully connected dense layer
	model.add(Flatten())
	# adds a fully connected layer
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	# adds the output layer with number of classes as 2 because there are 2 possible outcomes
	model.add(Dense(num_classes, activation='softmax'))
	
	model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

	print("Created CNN model")

	model.fit(data_train, labels_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(data_test, labels_test))

	print("Finished training CNN model")

	return model
	
resize_images()
data_train, data_test, labels_train, labels_test, input_shape = create_datasets()
model = train_model(input_shape)

score = model.evaluate(data_test, labels_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

