import cv2
import cv2.aruco as aruco
import numpy as np
import os
import ArucoModule as arm
import time

cap = cv2.VideoCapture(0)
# imgAug = cv2.imread("Markers/23.png")
augDics = arm.loadAugImages("Markers")


while True:
    secuess, img = cap.read()
    arucoFound = arm.findArucoMarkers(img)

    # loop through all the markers and augment each one
    if len(arucoFound[0]) != 0:
        for bbox, id in zip(arucoFound[0], arucoFound[1]):
            if int(id) in augDics.keys():
                img = arm.augmentAruco(bbox.astype(int), id, img, augDics[int(id)])

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    # print("time :", stop-start)