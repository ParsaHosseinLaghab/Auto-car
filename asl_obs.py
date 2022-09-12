import cv2
import numpy as np
import AVISEngine
from time import time
from time import sleep
n = 0

car = AVISEngine.car()
car.connect("127.0.0.1", 25001)
while car.getImage() is None:
    car.getData()
    sleep(1)

while True:
    car.setSpeed(100)
    car.getData()
    frame = car.getImage()
    frame = cv2.resize(frame, (512, 512))
    h, w = frame.shape[:2]

    black = [[[0, 0], [0, h], [int(w/5), h], [int(w/5), int(h*0.35)],
              [int(4*w/5), int(h*0.35)], [int(w*4/5), h], [w, h], [w, 0]]]
    external_poly = np.array(black, dtype = np.int32)
    # cv2.fillPoly(frame,external_poly,(80,80,80))

    black_car = [[[int(w/3), int(3*h/4)], [int(2.6*w/12), h],
            [int(4.2*w/5), h], [int(4.5*w/6), int(3*h/4)]]]
    external_poly = np.array(black_car,dtype=np.int32)
    cv2.fillPoly(frame,external_poly,(80,80,80))
    
    
    

    upper_gray = np.array([120,50,140])
    lower_gray = np.array([90,10,100])
    lower_white = np.array([0, 0, 168])
    upper_white = np.array([172, 111, 255])
    lower_yellow = np.array([22, 93, 0])
    upper_yellow = np.array([45, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (15, 15), 2)

    obstacles = cv2.inRange(hsv,lower_gray,upper_gray)
    white_line = cv2.inRange(hsv,lower_white,upper_white)
    yellow_line = cv2.inRange(hsv,lower_yellow,upper_yellow)


    y_white,x_white = np.where(white_line > 230)
    y_yellow,x_yellow = np.where(yellow_line > 230)

    try:
        a_white,b_white = np.polyfit(y_white,x_white,1)
        def white_fy(y) :  return a_white*y + b_white
    except:
        continue
    try:
        a_yellow,b_yellow = np.polyfit(y_yellow,x_yellow,1)
        def yellow_fy(y) : return a_yellow*y + b_yellow
    except:
        continue

    if w/2 >= yellow_fy(400):
        black = [[[0, 0], [w, 0], [w, h], [int(white_fy(h)), h], [int(white_fy(300)), 300], [int(white_fy(
            150)), 150], [int(yellow_fy(150)), 150], [int(yellow_fy(300)), 300], [int(yellow_fy(h)), h], [0, h]]]
        external_poly = np.array(black, dtype=np.int32)

    else:
        black = [[[0, 0], [w, 0], [w, h], [int(yellow_fy(h)), h], [int(yellow_fy(300)), 300], [int(yellow_fy(
            150)), 150], [int(white_fy(150)), 150], [int(white_fy(300)), 300], [int(white_fy(h)), h], [0, h]]]
        external_poly = np.array(black, dtype=np.int32)

    cv2.fillPoly(frame, external_poly, (80, 80, 80))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (15, 15), 2)

    obstacles = cv2.inRange(hsv,lower_gray,upper_gray)
    cv2.imshow("black",obstacles)



    if cv2.waitKey(10) == ord("q"):
        break

cv2.destroyAllWindows()
