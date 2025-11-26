import cv2
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "mods")
os.makedirs(output_dir, exist_ok=True)

drawing = False
roi_selected = False
x_start, y_start, x_end, y_end = -1, -1, -1, -1

color_lower = np.array([0, 0, 0])
color_upper = np.array([180, 255, 255])
color_mode = False 
binary_threshold = 70  
max_binary_value = 255  

# 鼠标回调函数
def draw_roi(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, drawing, roi_selected
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_start, y_start = x, y
        roi_selected = False
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            x_end, y_end = x, y
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_selected = True
        x_end, y_end = x, y

cap = cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)
cap.set(cv2.CAP_PROP_FPS, 30)
# cv2.namedWindow("Template Maker")
cv2.namedWindow("HSV Thresholds")
cv2.setMouseCallback("HSV Thresholds", draw_roi)

cv2.createTrackbar("H min","HSV Thresholds",0,180,lambda x:None)
cv2.createTrackbar("s min","HSV Thresholds",0,255,lambda x:None)
cv2.createTrackbar("v min","HSV Thresholds",0,255,lambda x:None)
cv2.createTrackbar("H max","HSV Thresholds",180,180,lambda x:None)
cv2.createTrackbar("S max","HSV Thresholds",255,255,lambda x:None)
cv2.createTrackbar("V max","HSV Thresholds",255,255,lambda x:None)
cv2.createTrackbar("Binary Threshold", "HSV Thresholds", binary_threshold, 255, lambda x: None)

print("menu")
print("1. choice ROI")
print("2. 's' save")
print("3. 'q' out")
print("4. 'c' change mode")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    b, g, r = cv2.split(frame) 
    # frame = cv2.merge([np.zeros_like(b), np.zeros_like(g), r])
    # frame = cv2.merge([np.zeros_like(b), g, np.zeros_like(r)])
    # frame = cv2.merge([b,g, np.zeros_like(r)])
    # frame = cv2.cvtColor(r_channel,cv2.COLOR_BGR2GRAY)

    h_min = cv2.getTrackbarPos("H min","HSV Thresholds")
    s_min = cv2.getTrackbarPos("S min","HSV Thresholds")
    v_min = cv2.getTrackbarPos("S min","HSV Thresholds")
    h_max = cv2.getTrackbarPos("H max","HSV Thresholds")
    s_max = cv2.getTrackbarPos("S max","HSV Thresholds")
    v_max = cv2.getTrackbarPos("V max","HSV Thresholds")
    binary_threshold = cv2.getTrackbarPos("Binary Threshold", "HSV Thresholds")

    color_lower = np.array([h_min, s_min, v_min])
    color_upper = np.array([h_max, s_max, v_max])

    if color_mode:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, binary_threshold, max_binary_value, cv2.THRESH_BINARY)

    else:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, color_lower, color_upper)
        gray = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        # _, binary = cv2.threshold(binary, 1, 255, cv2.THRESH_BINARY)
       

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY )
    # binary = binary[200:600, 150:1080]
    binary =cv2.GaussianBlur(gray,(5,5),0)
    display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    if drawing or roi_selected:
        cv2.rectangle(display, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    

    mode_text = "color mode:" if color_mode else "binary mode"
    cv2.putText(display, mode_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if color_mode:
        hsv_text = f"HSV: [{h_min},{s_min},{v_min}] - [{h_max},{s_max},{v_max}]"
        cv2.putText(display, hsv_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow("HSV Thresholds",display)
    
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and roi_selected:
        template_name = input("name: ")
        roi = binary[min(y_start, y_end):max(y_start, y_end), 
                     min(x_start, x_start):max(x_end, x_end)]
        if roi.size > 0:
            save_path = os.path.join(output_dir, f"{template_name}.png")
            cv2.imwrite(save_path, roi)
            print(f"save to: {save_path}")
            roi_selected = False
    
    elif key == ord('c'):
        color_mode = not color_mode
        print(f"color mode: {'start' if color_mode else 'end'}")
    
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()