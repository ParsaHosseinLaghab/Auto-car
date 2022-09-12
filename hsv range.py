import cv2
import numpy as np

def callback(x):
 pass

cv2.namedWindow('image')

ilowH = 100
ihighH = 160

ilowS = 0
ihighS = 100
ilowV = 0
ihighV = 75
blur = 0
# create trackbars for color change
cv2.createTrackbar('lowH','image',ilowH,179,callback)
cv2.createTrackbar('highH','image',ihighH,179,callback)

cv2.createTrackbar('lowS','image',ilowS,255,callback)
cv2.createTrackbar('highS','image',ihighS,255,callback)

cv2.createTrackbar('lowV','image',ilowV,255,callback)
cv2.createTrackbar('highV','image',ihighV,255,callback)
cv2.createTrackbar('blur','image',blur,255,callback)

def getthresholdedimg(hsv):
        threshImg = cv2.inRange(hsv,np.array((cv2.getTrackbarPos('Hue_Low','Trackbars'),cv2.getTrackbarPos('Saturation_Low','Trackbars'),cv2.getTrackbarPos('Value_Low','Trackbars'))),np.array((cv2.getTrackbarPos('Hue_High','Trackbars'),cv2.getTrackbarPos('Saturation_High' ,'Trackbars'),cv2.getTrackbarPos('Value_High','Trackbars'))))
        return threshImg


def getTrackValue(value):
        return value

#cap = cv2.VideoCapture('race.mp4')
img = cv2.imread("image.png")
while True:
    # grab the frame
#     ret, frame = cap.read()
#     if not ret:
#         cap = cv2.VideoCapture('race.mp4')
#         continue
    frame = img.copy()
    # get trackbar positions
    ilowH = cv2.getTrackbarPos('lowH', 'image')
    ihighH = cv2.getTrackbarPos('highH', 'image')
    ilowS = cv2.getTrackbarPos('lowS', 'image')
    ihighS = cv2.getTrackbarPos('highS', 'image')
    ilowV = cv2.getTrackbarPos('lowV', 'image')
    ihighV = cv2.getTrackbarPos('highV', 'image')

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (15, 15), cv2.getTrackbarPos('blur', 'image'))
    lower_hsv = np.array([ilowH, ilowS, ilowV])
    higher_hsv = np.array([ihighH, ihighS, ihighV])
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

    frame = cv2.bitwise_and(frame, frame, mask=mask)

    # show thresholded image
    cv2.imshow('hsv', hsv)
    cv2.imshow('mask', mask)
    cv2.imshow('image', frame)
    k = cv2.waitKey(100) & 0xFF # large wait time to remove freezing
    if k == 113 or k == 27:
        break
#cap.release()
cv2.destroyAllWindows()
