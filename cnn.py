from __future__ import print_function
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

original_parasitized = 'C:\\Users\\reach\\Documents\\Stuff\\Machine Learning\\Malaria-Detector\\original_cell_images\\Parasitized\\'
original_uninfected = 'C:\\Users\\reach\\Documents\\Stuff\\Machine Learning\\Malaria-Detector\\original_cell_images\\Uninfected\\'
resized_parasitized = 'C:\\Users\\reach\\Documents\\Stuff\\Machine Learning\\Malaria-Detector\\resized_cell_images\\Parasitized\\'
resized_uninfected = 'C:\\Users\\reach\\Documents\\Stuff\\Machine Learning\\Malaria-Detector\\resized_cell_images\\Uninfected\\'
original_images = 'C:\\Users\\reach\\Documents\\Stuff\\Machine Learning\\Malaria-Detector\\original_cell_images\\'
resized_images = 'C:\\Users\\reach\\Documents\\Stuff\\Machine Learning\\Malaria-Detector\\resized_cell_images\\'

batch_size, num_classes, epochs = 128, 2, 10
channels, img_rows, img_cols = 3, 32, 32
num_parasitized_images, num_uninfected_images = 13779, 13779

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs = {}):
        self.losses = []

    def on_batch_end(self, batch, logs = {}):
        self.losses.append(logs.get('loss'))

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
	data = np.ndarray(shape=(num_parasitized_images + num_uninfected_images, img_rows, img_cols, channels), dtype=np.float32)
	image_count = 0
	for filename in os.listdir(resized_parasitized):
		img = load_img(resized_parasitized + filename)
		img_array = img_to_array(img) # shape is (64, 64, 3)
		img_array.reshape((img_rows, img_cols, channels))
		data[image_count] = img_array
		image_count += 1
		if image_count % 1000 == 0:
				print("Gathering parasitized cell image " + str(image_count))

	print("Finished gathering parasitized cell images")

	for filename in os.listdir(resized_uninfected):
		img = load_img(resized_uninfected + filename)
		img_array = img_to_array(img) # shape is (64, 64, 3)
		img_array.reshape((img_rows, img_cols, channels))
		data[image_count] = img_array
		image_count += 1
		if image_count % 1000 == 0:
				print("Gathering uninfected image " + str(image_count))

	print("Finished gathering uninfected cell images")

	input_shape = (img_rows, img_cols, 3)

	labels = np.ndarray(shape = (num_parasitized_images + num_uninfected_images), dtype = int)
	for n in range(0, num_parasitized_images):
		labels[n] = 1
	for n in range(num_parasitized_images, num_parasitized_images + num_uninfected_images):
		labels[n] = 0
	
	print("Generated labels")

	labels = keras.utils.to_categorical(labels, num_classes)

	print("One-hot encoded labels")

	idx = np.random.permutation(len(data))
	data, labels = data[idx], labels[idx]

	print("Shuffled data and labels")

	data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size = 0.20)
	
	print("Split training and test sets")

	data_train /= 255
	data_test /= 255

	print("Normalized data")

	return data_train, data_test, labels_train, labels_test, input_shape

def train_model(input_shape):
	model = Sequential()
	model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = input_shape))
	model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
	# model.add(MaxPooling2D(pool_size = (2, 2)))
	# model.add(Dropout(0.25))
	model.add(Flatten())
	# model.add(Dense(128, activation = 'relu'))
	# model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation = 'softmax'))
	
	model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(), metrics = ['accuracy'])

	print("Created CNN model")

	# history = LossHistory()
	# model.fit(data_train, labels_train, batch_size = batch_size, epochs = epochs, verbose = 1, callbacks = [history])
	history = model.fit(data_train, labels_train, batch_size = batch_size, epochs = epochs, verbose = 1)
	
	print("Finished training CNN model")

	return model, history
	
def check(folder, image_name, model):
	data = np.ndarray(shape = (1, img_rows, img_cols, channels), dtype = np.float32)
	img, img_array = None, None
	if image_name in os.listdir(resized_images + folder):
		print(resized_images + folder + image_name)
		img = load_img(resized_images + folder + image_name)
		img_array = img_to_array(img) # shape is (64, 64, 3)
		img_array.reshape((img_rows, img_cols, channels))
		data[0] = img_array
		print(model.predict(data, verbose = 1))
	else:
		print(filename + " not found")

def save_cnn(model):
	model.save('cnn.h5')
	del model

def load_cnn():
	return load_model('cnn.h5')

#resize_images()
data_train, data_test, labels_train, labels_test, input_shape = create_datasets()
model, history = train_model(input_shape)
score = model.evaluate(data_test, labels_test, verbose = 1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""plt.plot(history.losses)
plt.title('Loss during Training')
plt.ylabel('Loss')
plt.xlabel('Batch')
plt.show()"""

plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()
save_cnn(model)
print("Model saved")

"""model = load_cnn()
print("Loaded model")

folder, filename = input("Parasitized or Uninfected?\n") + "\\", input("Filename?\n") + ".png"

while folder != "Done":
	check(folder, filename, model)
	folder = input("Parasitized or Uninfected?\n")
	filename = input("Filename?\n")"""



