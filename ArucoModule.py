import cv2
import cv2.aruco as aruco
import numpy as np
import os


def loadAugImages(path):
    """"
    :param path: folder in which all the marker images with ids are stored
    :return: dictionary with key as the id and values as the augmented image
    """
    myList = os.listdir(path)
    noOfMarkers = len(myList)
    print("Total Number of Markers Detected: ", noOfMarkers)
    augDics = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f'{path}/{imgPath}')
        augDics[key] = imgAug

    return augDics


def findArucoMarkers(img, markerSize=4, totalMarkers=100, draw=True):
    """"
    :param img: image in which to find the aruco markers
    :param markerSize: the size of the markers
    :param totalMarkers: total number of markers that compose the dictionary
    :param draw: flag to draw bbox around markers detected
    :return: bounding boxes and id numbers of markers detected
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)
    # print(ids)

    if draw:
        aruco.drawDetectedMarkers(img, bboxs)
    return [bboxs, ids]


def augmentAruco(bbox, id, img, imgAug, drawId=True):
    """
    :param bbox: the four corner points of the box
    :param id: marker id of the corresponding box used only for display
    :param img: the final image on wich to draw
    :param imgAug: the image that will overlapped on the marker
    :param drawId: flag to display the id of the detected markers
    :return: image with the augment image overlaid
    """
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    h, w, c = imgAug.shape

    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))
    imgOut = img + imgOut
    # print(tl)

    if drawId:
        cv2.putText(imgOut, str(id), tl,cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 255),2)

    return imgOut


def main():
    cap = cv2.VideoCapture(0)
    # imgAug = cv2.imread("Markers/23.png")
    augDics = loadAugImages("Markers")

    while True:
        secuess, img = cap.read()
        arucoFound = findArucoMarkers(img)

        # loop throug all the markers and augment each one
        if len(arucoFound[0]) != 0:
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                if int(id) in augDics.keys():
                    img = augmentAruco(bbox.astype(int), id, img, augDics[int(id)])

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
