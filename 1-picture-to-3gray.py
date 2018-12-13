#!/usr/bin/python

# -*- coding:utf-8 -*-


from PIL import Image


def read_pic(path):
	im=Image.open(path)
	return im

def resize_pic(image,width,height):
	im=image.resize((width,height))
	return im


def get_channels(image):
	imr,img,imb=image.split()s     
	'''相反情况是 im=Image.merge('RGB',(imr,img,imb))'''
	return imr,img,imb



if __name__=="__main__":
	'''读取图片，
	缩放图片，
	将一张RGB图像分为三个通道图像
	exercise with PIL'''
	path="D://照片/寝室毕业照/寝室毕业照IMG_20181202_115137.jpg"
	Im=read_pic(path)
	Im_resi=resize_pic(Im,256,256)
	Im_R,Im_G,Im_B=get_channels(Im_resi)

	#另外还有旋转语句  im.rotate(45)图片旋转45度，可用
	   #                im.save()   im.save(outfile, "JPEG")



'''note:
1、可以用help(Image)得到更多辅助信息，得到参考；
2、具体的操作示例可以参考Pillow的文档，网址：https://pillow.readthedocs.io/en/3.0.x/handbook/tutorial.html
'''