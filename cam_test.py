import cv2
import numpy as np
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"
cv2.namedWindow("Controls")

cap = cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)
# cap.set(cv2.CAP_PROP_FPS,30)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  

# # cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
# cap.set(cv2.CAP_PROP_EXPOSURE, 1) 
# cap.set(cv2.CAP_PROP_GAIN, 0) 
# cap.set(cv2.CAP_PROP_AUTO_WB, 0)
# exposure_value = 0  
# cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)      
# cv2.createTrackbar("Exposure", "Controls", 10, 30, lambda x: None) 

while True:
    # exposure_slider = cv2.getTrackbarPos("Exposure", "Controls")
    # exposure_value = exposure_slider - 23
    
    # cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value) 
    ret, frame = cap.read()
    if not ret:
        break


    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = (45, 45, 45) 
    upper = (255, 255, 255)
    mask = cv2.inRange(frame, lower, upper)
    # mask_inv = cv2.bitwise_not(mask)  

    white = np.full_like(frame, 255)           
    frame_ = np.where(mask[:, :, None] == 255, white, frame)
    # cv2.imshow("mask",mask)
    gray = cv2.cvtColor(frame_,cv2.COLOR_BGR2GRAY)
    _,binary = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)
    cv2.imshow("Controls", frame_)
    cv2.imshow("binary",frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

 
cv2.destroyAllWindows()