import cv2
import sys
import os

vidcap = cv2.VideoCapture(sys.argv[1])

def getFrame(sec, path):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(path + "/image"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames

sec = 0
frameRate = 0.5 #//it will capture image in each 0.5 second
count=1
path = './datasets/videoData/images'
os.makedirs(path, exist_ok=True)
success = getFrame(sec, path)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec, path)