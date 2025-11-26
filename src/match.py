from hmac import new
from xml.dom.expatbuilder import theDOMImplementation
import cv2
import os
import numpy as np
import math
from collections import deque

class TemplateMatcher:
    def __init__(self, template_dir="mods", rotation_angles=np.arange(-2, 2, 2), match_threshold=0.8, scale_factors=np.linspace(0.8, 1.1, 4)):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        the_dir = os.path.join(script_dir, template_dir)
        self.templates, self.template_info = self.load_templates(the_dir, rotation_angles)
        self.MATCH_THRESHOLD = match_threshold
        self.SCALE_FACTORS = scale_factors
        self.match_results = []
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.name_mapping = {}
        self.initialize_name_mapping()

    def initialize_name_mapping(self):
        for info in self.template_info:
            base_name = info['name']
            self.name_mapping[base_name] = base_name

    def add_name_mapping(self, original_name, display_name):
        self.name_mapping[original_name] = display_name

    def get_display_name(self, original_name):
        return self.name_mapping.get(original_name, original_name)

    def load_templates(self, template_dir, rotation_angles):
        templates = []
        template_info = []
        for file_name in os.listdir(template_dir):
            if file_name.endswith(".png"):
                path = os.path.join(template_dir, file_name)
                template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    for angle in rotation_angles:
                        rotated = self.rotate_image(template, angle)
                        templates.append(rotated)
                        template_info.append({
                            'name': os.path.splitext(file_name)[0],
                            'angle': angle
                        })
        print(f"load {len(templates)} mod")
        return templates, template_info

    def rotate_image(self, image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        rotation_matrix[0, 2] += (new_w - w) // 2
        rotation_matrix[1, 2] += (new_h - h) // 2
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), borderValue=255,flags=cv2.INTER_LINEAR)
        return rotated

    def non_max_suppression(self, results, overlap_thresh=0.8):
        if len(results) == 0:
            # print("no input")
            return []
        
        boxes = []
        for r in results:
            x, y = r['location']
            w, h = r['size']
            boxes.append([x, y, x + w, y + h, r['confidence']])
        
        boxes = np.array(boxes)
        # print(f"NMS start: {len(boxes)}")


        idxs = np.argsort(boxes[:, 4])[::-1]
        pick = []

        while len(idxs) > 0:
            i = idxs[0]
            pick.append(i)
            
            xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            intersection = w * h
            
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_j = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
            iou = intersection / (area_i + area_j - intersection)

            suppress_idx = np.where(iou > overlap_thresh)[0] + 1 
            idxs = np.delete(idxs, np.concatenate(([0], suppress_idx)))
        # print(f"NMS end: {len(pick)}")
        return [results[i] for i in pick]

    def process_frame(self, binary):
        self.match_results.clear()
        for i, template in enumerate(self.templates):
            info = self.template_info[i]
            orig_h, orig_w = template.shape
            best_match = None
            for scale in self.SCALE_FACTORS:
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                if new_w <= 0 or new_h <= 0 or new_w > binary.shape[1] or new_h > binary.shape[0]:
                    continue
                resized_template = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                res = cv2.matchTemplate(binary, resized_template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if max_val > self.MATCH_THRESHOLD:
                    if best_match is None or max_val > best_match['confidence']:
                        best_match = {
                            'name': info['name'],
                            'location': max_loc,
                            'size': (new_w, new_h),
                            'scale': scale,
                            'angle': info['angle'],
                            'confidence': max_val
                        }
            if best_match is not None:
                self.match_results.append(best_match)
        
        self.match_results = self.non_max_suppression(self.match_results)
        self.match_results.sort(key=lambda x: x['confidence'])


        return self.match_results
        
    
if __name__ == "__main__":
    cap=cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    cap.set(cv2.CAP_PROP_FPS,30)

    matcher_3 = TemplateMatcher( template_dir="mods",match_threshold=0.75)
    matcher_3.add_name_mapping("green1", "G.")
    matcher_3.add_name_mapping("green2", "G.")
    matcher_3.add_name_mapping("none1", "R.")
    matcher_3.add_name_mapping("none2", "R.")
    matcher_3.add_name_mapping("red1", "R.")
    matcher_3.add_name_mapping("red2", "R.")
    matcher_3.add_name_mapping("nopark", "P.")
    matcher_3.add_name_mapping("nopa", "P.")
    matcher_3.add_name_mapping("back", "B.")

    while True:
        ret,frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray=cv2.GaussianBlur(gray,(3,3),0)
            # # gray= gray[200:600, 460:780]

            results = matcher_3.process_frame(gray)
          
            cv2.imshow("m", gray)
            if results:
                # print("".join(result['name'] for result in results))  
                # print(matcher_3.get_display_name(results[0]['name']))
                best_result = max(results, key=lambda x: x['confidence'])
                print(best_result['name'])  
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()