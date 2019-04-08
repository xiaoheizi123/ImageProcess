import numpy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os






def images_to_video(save_path ,video_folder, rep=5, result_filename=None):
    if result_filename is None:
        result_filename = "{}.avi".format(save_path)
        #video_folder:文件目录
        # #os.listdir(video_folder):文件夹下边的图片列表
        # #f:0.jpg.........jpg
        # #splitext(f[0]）:把数字提取出来
        images_name = {int(os.path.splitext(f)[0]): os.path.join(video_folder, f) for f in os.listdir(video_folder)}
        # read the first frame and find the height, width and layers of all the images
        img = cv2.imread(images_name[0])
        height, width, layers = img.shape
        # initiate the video with width, height and fps = 25
        four_cc = cv2.VideoWriter_fourcc(*"XVID")# avi
        # four_cc = cv2.VideoWriter_fourcc(*"mp4v")  # mp4
        video = cv2.VideoWriter(result_filename, four_cc, 25, (width, height))
        for i in range(0, len(images_name)):
            for j in range(rep):
                img = cv2.imread(images_name[i])
                video.write(img)
            # print the progress bar
            if i % 100 == 0:
                print("Done {}%".format((i*100)/len(images_name)))
                cv2.destroyAllWindows()
                video.release()
                print("Done!")
                return None


if __name__ == '__main__':
    im_list=glob.iglob('data_zr/video_images'+ '/*.jpg')
    fig = plt.figure(figsize=[12, 12])
    for i, im_name in enumerate(im_list):
        im = cv2.imread(im_name)
        print( im_name.split('/')[2].split('.')[0])
        if not os.path.exists('../DensePoseData/infer_out/' + im_name.split('/')[2].split('.')[0]+'_IUV.png'):
            continue

        IUV = cv2.imread('../DensePoseData/infer_out/' + im_name.split('/')[2].split('.')[0]+'_IUV.png')
        INDS = cv2.imread('../DensePoseData/infer_out/' + im_name.split('/')[2].split('.')[0]+'_INDS.png', 0)
        #fig = plt.figure(figsize=[12, 12])
        plt.imshow(im[:, :, ::-1])
        plt.contour(IUV[:, :, 1] / 256., 10, linewidths=1)
        plt.contour(IUV[:, :, 2] / 256., 10, linewidths=1)
        plt.axis('off')

        plt.savefig("../DensePoseData/data_zr/detected/" + im_name.split('/')[2].split('.')[0] + ".png")
        plt.clf()
        #plt.cla()
        #plt.close("all")


