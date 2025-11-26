import time
import cv2 as cv
import numpy as np
import tensorflow as tf
from send_data import SerialPort

class LineTracker:
    def __init__(self, threshold=100, stride=3, range_num=10,black_ratio_threshold=0.02):
        self.THRESHOLD = threshold
        self.STRIDE = stride
        self.RANGE_NUM = range_num
        self.black_ratio_threshold = black_ratio_threshold
        self.order = ""

    def process_frame(self, binary):
        data = np.array(binary)
        data_after = tf.convert_to_tensor(data, tf.float32, name='data_after')
        data_after = tf.expand_dims(data_after, axis=0)
        data_after = tf.expand_dims(data_after, axis=3)
        out = tf.nn.max_pool(data_after, [1, self.STRIDE, self.STRIDE, 1], [1, self.STRIDE, self.STRIDE, 1], 'SAME')
        out = tf.nn.max_pool(out, [1, self.STRIDE, self.STRIDE, 1], [1, self.STRIDE, self.STRIDE, 1], 'SAME')
        out = tf.squeeze(out)
        pred = np.array(out, np.uint8)

        source = 0
        black_point_num = 0
        x = pred.shape[1]
        y = pred.shape[0]
        opt = 0
        if self.order == "F.":
            opt = int(x / 4)
        elif self.order == "R.":
            opt = int(x / 3)
        else:
            opt = 0
        for i in range(y):
            row = 0
            row_white_point_num = 0
            for j in range(opt, x):
                if pred[i, j] == 0:
                    row += 1
                    row_white_point_num = 0
                    source += j
                else:
                    row_white_point_num += 1
                    if row > 5 and row_white_point_num > 5:
                        break
            black_point_num += row

            
        total_pixels = pred.size
        black_ratio = black_point_num / total_pixels

        if black_ratio < self.black_ratio_threshold:
            result = "S."
            self.order = "S."

        else:
            source /= black_point_num
            source -= x / 2
            if abs(source) <= self.RANGE_NUM:
                result = "F."
            elif source > 0:
                result = "R."
            else:
                result = "L."
            self.order = result
        return result
    
    def filter_color_to_white(self,bgr_img,
                          lower_rgb=(0, 0, 0),   
                          upper_rgb=(255, 255, 255)):
        lower_bgr = lower_rgb[::-1]
        upper_bgr = upper_rgb[::-1]
        mask = cv.inRange(bgr_img, lower_bgr, upper_bgr)  
        res  = bgr_img.copy()
        res[mask > 0] = (255, 255, 255)                  
        return res
    
if __name__ == '__main__':

    serial_port = SerialPort(
        port='/dev/ttyAMA0',
        baudrate=115200,
        send_format='ascii',
        recv_format='ascii'
    )
    cap = cv.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    cap.set(cv.CAP_PROP_FPS,30) 

    lower = (50, 70, 60) 
    upper = (255, 255, 255)

   


    if not cap.isOpened():
        raise RuntimeError('Cannot open camera')

    tracker = LineTracker(threshold=100, stride=3, range_num=10, black_ratio_threshold=0.03)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = tracker.filter_color_to_white(frame, lower, upper)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray, tracker.THRESHOLD, 255, cv.THRESH_BINARY)

        cmd = tracker.process_frame(binary)
        # print(cmd)     
        serial_port.send_data(cmd)         

        cv.imshow('frame',frame)
        # cv.imshow('binary', binary)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
