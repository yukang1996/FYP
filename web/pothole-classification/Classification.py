#Importing Keras libraries and packages

"""
Process of building a CNN always involve 4 major steps.
1)	Convolution
2)	Pooling
3)	Flattening
4)	Full connection
"""
import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import *
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential #initialise our NN model as sequantial. There are 2 basic ways of initailising: sequence or graph
from keras.layers import Conv2D #Step-1: perform convolution operation. 2D for image, 3D for video
from keras.layers import MaxPooling2D #Step-2: pooling operation.MaxPooling, we need maximum value pixel from respective region of interest
from keras.layers import Flatten #Step-3: flattening, convert all resultant 2D arrays into a single long continuous linear vector
from keras.layers import Dense #Step-4: perfrom full connection of NN.
import tensorflow as tf
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import colorsys as cs


def _init_(graph, sess):
	global new_model,training_set
	with graph.as_default():
		with sess.as_default():
			classifier = Sequential() #declare object with sequential 

			"""
			add convolution layer. Conv2D has 4 arg:
			-number of filters:32
			-shape of filter:3x3
			-input shape & type of image:64x64 and 3 stand for rgb
			-activation function: relu
			64-32+2(0) / 1 
			"""
			classifier.add(Conv2D(32,(3,3),input_shape = (64, 64, 3),activation = 'relu'))

			"""
			perform pooling operation on resultant feature maps after convolution operation is done
			on image. The aim of pooling operation is to reduce the size of images as much as possible.
			We are trying to reduce total number of nodes for upcoming layers.

			add pooling layer to classifier object. 
			we take pooling size of 2x2 so that we will have minimum pixel loss & get precise region
			where feature are located.
			We reduced complexity of model without reducing its performance
			32/2 
			"""
			classifier.add(MaxPooling2D(pool_size = (2,2)))

			"""
			convert pooled image into continouous vector through flattening. Taking 2D array,
			pooled image pixels and converting them to 1D single vector

			"""
			classifier.add(Flatten())

			"""
			Create fully connected layer. We are going to connect set of nodes we got after flattening
			step, these nodes will act as an input layer to these fully connected layers. This layer
			will be present between input layer & output layer( hidden layer )

			Dense is function to add a fully connected layer. 
			-units is number of nodes that should be present in this hidden layer, these unit value will
			be always between number of input nodes & output nodes. Most optimal number of nodes 
			can be achieved through experimental tries. Common practice to use power of 2
			-activation function = rectifier function
			"""
			classifier.add(Dense(units = 128, activation = 'relu'))

			"""
			Initialise output layer which should contain only 1 node as it is binary classification.
			Single node will give us binary output of either Cat or Dog

			Final layer contains only 1 node. Activation function is sigmoid

			"""
			classifier.add(Dense(units = 1, activation = 'sigmoid'))

			"""
			optimizer parameter is to choose stochastic gradient descent algorithm
			loss parameter is to chose loss function
			metrics parameter is to choose performance metric
			"""
			adam = optimizers.Adam(lr = 0.001)
			classifier.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy'])

			"""
			preprocess the image to prevent overfitting
			Overfitting is when you get a great training accuracy and very poor test accuracy 
			due to overfitting of nodes from 1 layer to another.

			So, we perform image augmentations which is synthesising training data. 
			Use keras.preprocessing library for synthesising part as well as to prepare training set 
			and test set.

			ImageDataGenerator- generates batches of tensor image data with real time data
			augmentation. The data will be looped over(in batches):
			-rescale: rescaling factor. multiply the data by value provided
			-shear_range: shear intensity(shear angle in counter-clockwise direction in degrees)
			-zoom_range: range for random zoom
			-horizontal_flip: randomly flip inputs horizontally

			"""

			train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
			test_datagen = ImageDataGenerator(rescale = 1./255)
			training_set = train_datagen.flow_from_directory(r'C:\Users\User\Desktop\django tutorial\fyp\web\pothole-classification\training_set', target_size = (64,64), batch_size = 100, class_mode = 'binary')
			test_set = train_datagen.flow_from_directory(r'C:\Users\User\Desktop\django tutorial\fyp\web\pothole-classification\test_set', target_size = (64,64), batch_size = 100, class_mode = 'binary')

			"""
			steps_per_epoch holds number of training images
			epochs is a single step in training neural network. when neural network is trained 
			on every training sample only in 1 pass we say that one epoch is finished.

			"""
			# classifier.fit_generator(training_set,steps_per_epoch = 8000 / 32,epochs = 25, validation_data = test_set, validation_steps = 2000 / 32)
			# classifier.save('training_result.h5')

			new_model = load_model(r'C:\Users\User\Desktop\django tutorial\fyp\web\pothole-classification\training_result.h5')

def detect_level_4(crop_image):
	image = cv2.resize(crop_image, (300, 300))
	median = cv2.medianBlur(image, 7)
	# cv2.imshow('blur', median)
	lower_range = np.array([0,0,0])
	upper_range = np.array([15,15,15])

	mask = cv2.inRange(median, lower_range, upper_range)
	# cv2.imshow('mask', mask)
	con_image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	area = 0
	if len(contours) != 0:
		cv2.drawContours(image, contours, -1, 255, 3)

		c = max(contours, key = cv2.contourArea )

		print(c)

		area = cv2.contourArea(c)
		print('area', area)
		x, y, w, h = cv2.boundingRect(c)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	else:
		print('no black detected')
		area = 0

	# cv2.imshow('detection', image)
	# cv2.waitKey(0)
	threshold = (300 * 300) * 0.03
	print('threshold', threshold)
	if(area < threshold):
		print('not level 4')
		return False;
	else:
		print('level 4')
		return True


def detect_level_1(graph, sess, crop_image):
	with graph.as_default():
		with sess.as_default():
			test_image = image.load_img(crop_image,target_size = (64,64))
			print(test_image)
			test_image = image.img_to_array(test_image)
			test_image = np.expand_dims(test_image,axis = 0)
			result = new_model.predict(test_image)
			print(result)
			training_set.class_indices
			print(training_set.class_indices)
			print(result)
			# max_num = np.argmax(result)
			# print('max', max_num)
			if result <= 0.5:
				prediction = 'crack'
				return True
			else:
				prediction = 'not crack'
				return False


def find_histogram(clt):
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	# bins is the x-axis value
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
	# convert value to float
	hist = hist.astype("float")

	# value of each column divide by sum of all value
	hist /= hist.sum()

	return hist

def plot_colors2(hist, centroids):
	print(centroids)
	bar = np.zeros((50, 300, 3), dtype = 'uint8')
	startX = 0

	print(list(zip(hist, centroids)))
	results = []
	for(percent, color) in zip(hist, centroids):
		results.append([percent, color])
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype('uint8').tolist(), -1) 
		startX = endX

	return bar, results

def verify_pothole(results):
	# 128 128 128
	print('verifyinnggg')
	print(results)
	
	final_h = 0.0
	final_s = 0.0
	final_v = 0.0
	depth_score = 0
	for i in results:
		print('x')
		print(i)
		area = i[0]
		r_temp = i[1][0] / 255
		g_temp = i[1][1] / 255
		b_temp = i[1][2] / 255
		print(r_temp, g_temp, b_temp)
		h, s, v = cs.rgb_to_hsv(r_temp, g_temp, b_temp)
		print('h = %s, s = %s, v = %s'%(h, s, v))
		temp_depth_score = (1 - s) + (1 - v) * 2 
		print('temp_depth_score', temp_depth_score)
		if(temp_depth_score > depth_score):
			depth_score = temp_depth_score


	print('xxxxxxxxxxxxxxxxxxxxxxxxxx')
	print(depth_score)
	
	depth_score /= 3 #normalise
	return depth_score

def predict_level(graph, sess, crop_image, original_image, trigger):
	if(trigger == True):
		isLevel1 = detect_level_1(graph, sess, crop_image)
		if(isLevel1 == True):
			prediction = 'Level 1'
			return prediction
		else:
			prediction = 'Nothing detected'
			return prediction

	image = cv2.imread(crop_image)
	original_image = cv2.imread(original_image)
	# cv2.imshow('ori', original_image)
	# cv2.imshow('crop', image)
	# cv2.waitKey(0)
	# detect level 4 ( black )
	isLevel4 = detect_level_4(image.copy())
	if(isLevel4 == True):
		prediction = 'Level 4'
		return prediction
	isLevel1 = detect_level_1(graph, sess, crop_image)
	if(isLevel1 == True):
		prediction = 'Level 1'
		return prediction

	n_cluster = 3
	img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	img = img.reshape((img.shape[0] * img.shape[1],3))
	clt = KMeans(n_cluster)
	clt.fit(img)
	hist = find_histogram(clt)
	bar, color = plot_colors2(hist, clt.cluster_centers_)
	# plt.axis('off')
	# plt.imshow(bar)
	# plt.show()
	depth_score = verify_pothole(color)
	print('depth: ', depth_score)
	print('image size', image.size)
	print('original size', original_image.size)
	ratio = image.size / original_image.size
	print('ratio:', ratio)

	final_ratio = ratio + depth_score
	print('final ratio', final_ratio)
	if(final_ratio <= 1.0):
		prediction = 'level 2'
	else:
		prediction = 'level 3'
	print(prediction)
	return prediction

		

		



