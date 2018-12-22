from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 
from torchvision.transforms import ToPILImage
import glob
import os


def flip(img):
	flip_img = np.fliplr(img)
	# flip_img = Image.fromarray(flip_img)
	return flip_img

def trans_l(img, trans):
	trans_img = np.zeros(img.shape)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if i < img.shape[0]-trans:
				trans_img[i][j] = img[i+trans][j]
	return trans_img

def trans_r(img, trans):
	trans_img = np.zeros(img.shape)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if i >= trans:
				trans_img[i][j] = img[i-trans][j]
	return trans_img

def trans_d(img, trans):
	trans_img = np.zeros(img.shape)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if j < img.shape[0]-trans:
				trans_img[i][j] = img[i][j+trans]
	return trans_img

def trans_u(img, trans):
	trans_img = np.zeros(img.shape)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if j > trans:
				trans_img[i][j] = img[i][j-trans]
	return trans_img

import imutils

def rotate(img, deg):
	rot_img = imutils.rotate(img, deg)
	rot_img = np.array(rot_img)
	return rot_img



os.chdir('training_set/Train1')
for file in glob.glob('*.png'):
	img = np.array(Image.open(file))
	# print(img)
	flip_img = flip(img)
	transl_img = trans_l(img, 20)
	transr_img = trans_r(img, 20)
	transu_img = trans_u(img, 20)
	transd_img = trans_d(img, 20)
	rot_img = rotate(img, 15)
	# trans_img
	# alpha = Image.fromarray(transl_img.astype('uint8'))
	aug_data = {0:flip_img, 1:transl_img, 2:transr_img, 3:transu_img, 4:transd_img, 5:rot_img}
	aug_img = {x: Image.fromarray(aug_data[x].astype('uint8')) for x in range(6)}
	aug_img = {x: aug_img[x].convert("L") for x in range(6)}
	os.chdir('../Augmented-Train1')
	aug_img[0].save("flip"+file)
	aug_img[1].save("transl"+file)
	aug_img[2].save("transr"+file)
	aug_img[3].save("transd"+file)
	aug_img[4].save("transu"+file)
	aug_img[5].save("rot"+file)
	os.chdir('../Train1')
