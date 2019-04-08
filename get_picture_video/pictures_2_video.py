import numpy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

from PIL import Image


def jpg2video(sp, fps):
    """ 将图片合成视频. sp: 视频路径，fps: 帧率 """
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    images = os.listdir('data_zr/detected')
    #images.sort(key = lambda x:int(x[12:-4]))
    images.sort(key=lambda x: int(x[:-4]))
    im = Image.open('data_zr/detected/' + images[0])
    vw = cv2.VideoWriter(sp, fourcc, fps, im.size)
 
    os.chdir('data_zr/detected')
    for image in range(len(images)):
        jpgfile = str(image + 1) + '.png'
        #jpgfile = 'video_images'+str(image + 1) + '.png'
        try:
            frame = cv2.imread(jpgfile)
            vw.write(frame)
        except Exception as exc:
            print(jpgfile, exc)
    vw.release()
    print(sp, 'Synthetic success!')
 
 
if __name__ == '__main__':
    
    sp_new = 'data_zr/video/zr_1.avi'
    
    jpg2video(sp_new, 28)  # 图片转视频

    #imgs=glob.glob('data_zr/detected/*.png')


    #images_to_video(os.path.join('../DensePoseData/data_zr/video/', 'zr_1'), '../DensePoseData/data_zr/detected/', 1)

