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
		img = load_img(original_parasitized + filename, target_size=(img_rows, img_cols))
		img_array = img_to_array(img)
		save_img(resized_parasitized + filename, img_array)
		image_count += 1
		if image_count % 1000 == 0:
			print("Resizing parasitized image " + str(image_count))

	image_count = 0
	for filename in os.listdir(original_uninfected):
		img = load_img(original_uninfected + filename, target_size=(img_rows, img_cols))
		img_array = img_to_array(img)
		save_img(resized_uninfected + filename, img_array)
		image_count += 1
		if image_count % 1000 == 0:
			print("Resizing uninfected image " + str(image_count))

def preprocess_images():
	input_shape = None
	# the data, split between train and test sets
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# shape[0] is number of images, 0 because original shape was (number of images, row pixels, column pixels) or (60000, 28, 28), specifying channels
	if K.image_data_format() == 'channels_first':
	    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	    input_shape = (1, img_rows, img_cols)
	else:
	    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	    input_shape = (img_rows, img_cols, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	# normalize values to be in [0, 1]
	x_train /= 255
	x_test /= 255
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	# class vector starts such that each image has a single array representing the probability of being each digit
	# changes so that instead of having dimensions (60000), it has dimensions (60000, 10) by splitting the array into classes of digits
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	return x_train, x_test, y_train, y_test, input_shape

def create_model():
	model = Sequential()

	# input shape is shape of one sample, or (1, 28, 28), no 60000 because it is one sample
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
	# adds the output layer with number of classes as 10 because there are 10 possible digits
	model.add(Dense(num_classes, activation='softmax'))

	return model

resize_images()
"""model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])"""
