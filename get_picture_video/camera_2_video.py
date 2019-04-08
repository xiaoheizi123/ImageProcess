##!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
c=time.time()
t=time.ctime(c)

cameraCapture = cv2.VideoCapture(0)
fps=30
videoname='video_from_camera/'+str(t)+'.avi'
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
size=(int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out=cv2.VideoWriter(videoname,fourcc,fps,size)
#cap=cv2.VideoCapture(0)
while(1):
    ret,frame=cameraCapture.read()   #get frame
    cv2.imshow("zhengr",frame)
    out.write(frame)
    if cv2.waitKey(1)&0xFF==ord('q')or ret==False:
        break
cameraCapture.release()
cv2.destroyAllWindows()

