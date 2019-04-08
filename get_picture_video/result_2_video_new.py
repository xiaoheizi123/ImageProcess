import numpy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from PIL import Image

def splited_2_pictures(imgpath,savepath):
    im_list = glob.iglob('data_zr/video_images' + '/*.jpg')
    fig = plt.figure(figsize=[12, 12])
    for i, im_name in enumerate(im_list):
        im = cv2.imread(im_name)
        print(im_name.split('/')[2].split('.')[0])
        if not os.path.exists('../DensePoseData/infer_out/' + im_name.split('/')[2].split('.')[0] + '_IUV.png'):
            continue

        IUV = cv2.imread('../DensePoseData/infer_out/' + im_name.split('/')[2].split('.')[0] + '_IUV.png')
        INDS = cv2.imread('../DensePoseData/infer_out/' + im_name.split('/')[2].split('.')[0] + '_INDS.png', 0)
        # fig = plt.figure(figsize=[12, 12])
        plt.imshow(im[:, :, ::-1])
        plt.contour(IUV[:, :, 1] / 256., 10, linewidths=1)
        plt.contour(IUV[:, :, 2] / 256., 10, linewidths=1)
        plt.axis('off')

        plt.savefig("data_zr/detected/" + im_name.split('/')[2].split('.')[0] + ".png")
        plt.clf()


def pictures_2_video(picpath,videopath,fps):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    images = os.listdir('data_zr/detected')
    images.sort(key=lambda x: int(x[12:-4]))
    im = Image.open('data_zr/detected'+'/' + images[0])
    vw = cv2.VideoWriter(videopath, fourcc, fps, im.size)

    os.chdir('data_zr/detected')
    for image in range(len(images)):
        # Image.open(str(image)+'.jpg').convert("RGB").save(str(image)+'.jpg')
        jpgfile = 'video_images' + str(image + 1) + '.png'
        try:
            frame = cv2.imread(jpgfile)
            vw.write(frame)
        except Exception as exc:
            print(jpgfile, exc)
    vw.release()
    print(videopath, 'Synthetic success!')




if __name__ == '__main__':

    splited_2_pictures('data_zr/video_images','data_zr/detected/')
    pictures_2_video('data_zr/detected','data_zr/video/zr_1.avi',28)



