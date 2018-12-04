#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import tensorflow as tf
import json
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import colorsys as cs

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


config_path  = r'C:\Users\User\Desktop\django tutorial\fyp\web\potholes-detection-master\config.json'
weights_path = r'C:\Users\User\Desktop\django tutorial\fyp\web\potholes-detection-master\trained_wts.h5'

with open(config_path) as config_buffer:    
    config = json.load(config_buffer)

###############################
#   Make the model 
###############################
def _init_(graph, sess):
	global yolo
	with graph.as_default():
		with sess.as_default():
			yolo = YOLO(backend         = config['model']['backend'],
		            input_size          = config['model']['input_size'], 
		            labels              = config['model']['labels'], 
		            max_box_per_image   = config['model']['max_box_per_image'],
		            anchors             = config['model']['anchors'])

		###############################
		#   Load trained weights
		############################### 
			yolo.load_weights(weights_path)

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
	final_area = 0.0
	for i in results:
		print('x')
		print(i)
		area = i[0]
		h_temp = i[1][0] / 255
		s_temp = i[1][1] / 255
		v_temp = i[1][2] / 255
		print(h_temp, s_temp, v_temp)
		h, s, v = cs.rgb_to_hsv(h_temp, s_temp, v_temp)
		print('hsv')
		print(h, s, v)
		if(s <= 0.6 and v <= 0.6):
			print('grey')
			final_area += area

	print('xxxxxxxxxxxxxxxxxxxxxxxxxx')
	print(final_area)
	return final_area



def _main_(graph, sess, input):
	trigger = False
	###############################
	#   Predict bounding boxes 
	###############################
	with graph.as_default():
		with sess.as_default():
			image = cv2.imread(input)
			tempo = image.copy()
			boxes = yolo.predict(image)
			image, crop_list = draw_boxes(image, boxes, config['model']['labels'])

			print(len(boxes), 'boxes are found')

			if(len(boxes) == 0):
				print('crack?')
				trigger = True
				cv2.imwrite(input[:-4] + '_detected' + input[-4:], image)
				return trigger,input

			cv2.imwrite(input[:-4] + '_detected' + input[-4:], image)
			print(len(crop_list))
			temp_area = 0
			temp = ''
			counter = 1
			n_cluster = 3
			for i in crop_list:
				print(i)
				xmin = i[0]
				ymin = i[1]
				xmax = i[2]
				ymax = i[3]
				crop_image = tempo[ymin:ymax, xmin:xmax]
				# cv2.imshow('crop', crop_image)
				# cv2.waitKey(0)
				crop_image_location = input[:-4] + '_cropped_'+ str(counter) + input[-4:]
				cv2.imwrite(crop_image_location, crop_image)
				counter += 1
				print('crop_image',crop_image)
				img = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
				img = img.reshape((img.shape[0] * img.shape[1],3))
				clt = KMeans(n_cluster)
				clt.fit(img)
				hist = find_histogram(clt)
				bar, color = plot_colors2(hist, clt.cluster_centers_)
				# plt.axis('off')
				# plt.imshow(bar)
				# plt.show()
				final_area = verify_pothole(color)
				if final_area > temp_area:
					temp_area = final_area
					temp = crop_image_location
	print('-------------------------------------------------------------------')
	print(temp)
	return trigger,temp

if __name__ == '__main__':
    _main_(image)
