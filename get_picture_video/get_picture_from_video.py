# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time


# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)




def get_frame(videoPath, svPath):
    cap = cv2.VideoCapture(videoPath)
    numFrame = 0
    while True:
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                #cv2.imshow('video', frame)
                numFrame += 1
                newPath = svPath + str(numFrame) + ".jpg"
                cv2.imencode('.jpg', frame)[1].tofile(newPath)
        if cv2.waitKey(10) == 27:
            break

def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    get_frame('./../DensePoseData/data_zr/VID_20190408_092158.mp4', './../DensePoseData/data_zr/video_images')

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    for i, im_name in enumerate(im_list):
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.pdf')
        )
        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )

        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            cls_bodys,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2
        )




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
    get_frame('video_from_camera/Mon Apr  8 15:56:54 2019.avi', './../DensePoseData/data_zr/video_images/')
#    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
#    setup_logging(__name__)
#    args = parse_args()
#    main(args)
#    images_to_video(os.path.join('../DensePoseData/data_zr/video/', 'zr_1'), '../DensePoseData/data_zr/detected/', 1)

