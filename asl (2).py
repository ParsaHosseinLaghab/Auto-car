import cv2
import numpy as np
import AVISEngine
from time import sleep
from time import time
# video = cv2.VideoCapture("race.avi")
car = AVISEngine.car()
car.connect("127.0.0.1", 25001)
while car.getImage() is None:
    car.getData()
    sleep(1)


x = 0
d = 0
error = 0
new_error = 0
while True:
    car.setSpeed(100)
    car.getData()
    frame = car.getImage()
    t = time()

    frame = cv2.resize(frame, (512, 512))
    copy = frame.copy()
    h, w = frame.shape[:2]
    h1 = int(0.3*h)
    copy_obs = frame.copy()

    # ---------------------------------------first making black--------------------
    black = [[[0, 0], [0, 0.65*h], [w/7, 0.65*h],
              [int(3.5*w/7), int(1.4*h1)], [int(4*w/7), int(1.4*h1)], [w, 0.65*h], [w, 0]]]
    external_poly = np.array(black, dtype=np.int32)
    cv2.fillPoly(copy, external_poly, (80, 80, 80))
    # cv2.fillPoly(copy_obs, external_poly, (80, 80, 80))
              
    # ------------------------------------------car black------------------------
    black = [[[int(w/3), int(3*h/4)], [int(2.6*w/12), h],
              [int(4.2*w/5), h], [int(4.5*w/6), int(3*h/4)]]]
    external_poly = np.array(black, dtype=np.int32)
    cv2.fillPoly(copy, external_poly, (80, 80, 80))
    cv2.fillPoly(copy_obs, external_poly, (80, 80, 80))
    # ------------------------------------------------------------------------------

    # ----------------------------color ranges ----------------------
    lower_blue = np.array([100, 0, 0])
    upper_blue = np.array([160, 100, 75])

    upper_gray = np.array([140,65,180])
    lower_gray = np.array([90,10,100])

    lower_yellow = np.array([22, 93, 0])
    upper_yellow = np.array([45, 255, 255])

    lower_white = np.array([0, 0, 168])
    upper_white = np.array([172, 111, 255])
    # --------------------------------------------------------------

    hsv = cv2.cvtColor(copy, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (15, 15), 2)
    hsv_obs = cv2.cvtColor(copy_obs, cv2.COLOR_BGR2HSV)
    hsv_obs = cv2.GaussianBlur(hsv_obs, (15, 15), 2)

    # ------ cv2.inrange will make the given range white
    obstacles = cv2.inRange(hsv_obs,lower_gray,upper_gray)
    yellow_line = cv2.inRange(hsv, lower_yellow, upper_yellow)
    white_line = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    cv2.imshow("obstacles", mask)


    # ----------- finding equations --------------------------
    yy, xx = np.where(yellow_line > 240)
    try:
        A_yellow, B_yellow = np.polyfit(yy, xx, 1)
        def yello_fx(x): return (A_yellow*x + B_yellow)
    except:
        continue

    yy1, xx1 = np.where(white_line > 240)
    try:
        A1_white, B1_white = np.polyfit(yy1, xx1, 1)
        def white_fx(x): return (A1_white*x + B1_white)
    except:
        continue
    # ----------------------------------------------------------
    # ----- findcontours return a list that have contours point lists in it -----
    points, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_points = sorted(points, key=len)

    try:
        lane1 = cv2.fillPoly(np.zeros_like(mask), pts=[
            sorted_points[-1]], color=(255))
        lane2 = cv2.fillPoly(np.zeros_like(mask), pts=[
            sorted_points[-2]], color=(255))
    except:
        # print(Exception)
        continue

    lane_pose_1 = np.mean(np.where(lane1[300:400, :] > 0), axis=1)[1]
    lane_pose_2 = np.mean(np.where(lane2[300:400, :] > 0), axis=1)[1]

    if abs(w/2-lane_pose_1) < abs(w/2-lane_pose_2):
        lane_pose_3 = lane_pose_1
    else:
        lane_pose_3 = lane_pose_2
    # ----------------------------------------------------------
    print(lane_pose_3)
    distance_yellow = w/2 - yello_fx(lane_pose_3)

    if (distance_yellow > 0):
        black = [[[0, 0], [w, 0], [w, h], [int(white_fx(h)), h], [int(white_fx(300)), 300], [int(white_fx(
            200)), 200], [int(yello_fx(200)), 200], [int(yello_fx(300)), 300], [int(yello_fx(h)), h], [0, h]]]
        # print(black)
        external_poly = np.array(black, dtype=np.int32)
    else:
        black = [[[0, 0], [w, 0], [w, h], [int(yello_fx(h)), h], [int(yello_fx(300)), 300], [int(yello_fx(
            200)), 250], [int(white_fx(200)), 200], [int(white_fx(300)), 300], [int(white_fx(h)), h], [0, h]]]
        external_poly = np.array(black, dtype=np.int32)

    cv2.fillPoly(mask, external_poly, (80, 80, 80))
    # cv2.fillPoly(obstacles)

    if not np.isnan(new_error and lane_pose_3):
        error = new_error
        new_error = lane_pose_3-w/2

    if not np.isnan(lane_pose_3):
        cv2.circle(frame, (int(lane_pose_3), 370), 15, (0, 150, 255), -1)
        x += new_error
        dt = 0.001/abs((((0.77*h)-370)/(new_error))) if abs((((0.77*h)-370)/(new_error)))> 1 else abs((((0.77*h)-370)/(new_error)))/1000
        d = (new_error-error)
        car.setSteering(0.6*(new_error)+dt*(x)+1.40625*(d))

    # if (car.getSensors()[0]<1450 or car.getSensors()[1]<1450 or car.getSensors()[2]<1450) and distance_yellow > 0:
    #     car.setSteering(-70)

    # elif (car.getSensors()[0]<1450 or car.getSensors()[1]<1450 or car.getSensors()[2]<1450) and distance_yellow < 0 :
    #     car.setSteering(70)
        
    # cv2.imshow("copy", copy)
    # cv2.imshow("hsv",hsv)
    # cv2.imshow('lane1',lane1)
    # cv2.imshow('lane2',lane2)
    # cv2.imshow('frame',frame)
    # cv2.imshow('frame1',yellow_line)
    # cv2.imshow('frame2',white_line

    # (lane_pose_1)
    # print(lane_pose_2)

    # cv2.imshow("mask", mask)
    if cv2.waitKey(10) == ord("q"):
        break

cv2.destroyAllWindows()
