import cv2
import os
import numpy as np

import cv2
import numpy as np
import os

class TreeSVMClassifier:
    def __init__(self, model_path):

        if not os.path.exists(model_path):
            raise FileNotFoundError(f'no find: {model_path}')
        self.model = cv2.ml.SVM_load(model_path)
        self.roi_size = (128, 256)  
        
    def predict(self, frame):

        h, w = frame.shape[:2]
        x0 = (w - self.roi_size[0]) // 2
        y0 = (h - self.roi_size[1]) // 2
        
        if x0 < 0 or y0 < 0 or (x0 + self.roi_size[0]) > w or (y0 + self.roi_size[1]) > h:
            return "Unknown"
        
        roi = frame[y0:y0+self.roi_size[1], x0:x0+self.roi_size[0]]
        
        blob = cv2.resize(roi, self.roi_size).astype(np.float32)
        blob = blob.reshape(1, -1)

        _, pred = self.model.predict(blob)
        label = int(pred[0, 0])

        if label == 0:
            return "red"
        elif label == 1:
            return "green"
        elif label == 2:
            return "none"
        else:
            return "Unknown"
        
if __name__ == "__main__":

    model_path = r'/home/hmkk/car_ws/tree_model.xml'
    classifier = TreeSVMClassifier(model_path)
    

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(3, 640)
    cap.set(4, 480)
    
    if not cap.isOpened():
        raise RuntimeError('cannot open USB camera')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print('get no frame')
            break

        frame = frame[0:480, 150:490]
        
        result = classifier.predict(frame)
        
        h, w = frame.shape[:2]
        x0 = (w - classifier.roi_size[0]) // 2
        y0 = (h - classifier.roi_size[1]) // 2
        
        cv2.rectangle(frame, (x0, y0), 
                     (x0 + classifier.roi_size[0], y0 + classifier.roi_size[1]), 
                     (0, 255, 0), 2)
        cv2.putText(frame, result, (x0, y0-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Tree Classification', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()