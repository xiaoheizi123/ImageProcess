#!/usr/bin/python

# -*- coding:utf-8 -*-

import cv2

def pic(path):
	im=cv2.imread(path)
	cv2.imshow('image',im)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return im
	
def pic_to_gray(image):
	#Applying Grayscale filter to image 作用Grayscale（灰度）过滤器到图像上
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return gray

def pic_gray_save(image):
	#保存过滤过的图像到新文件中
	cv2.imwrite('graytest.jpg',image)


if __name__ == '__main__':
	'''opencv版的显示图像，以及转化为灰度（因为使用的是滤波器，直接就变成单通道了）
	具体的做法现在不做追究，
	读取、显示图像，
	转化为灰度，
	保存
	'''
	path="D://test.jpg"     #注意用opencv读取图像的时候路径不能有中文
	im=pic(path)
	im_gray=pic_to_gray(im)
	pic_gray_save(im_gray)