import cv2
import threading
import queue
from send_data import SerialPort
from match import TemplateMatcher
from detect import TreeSVMClassifier
from track import LineTracker
from LLM_Module import LLMProcessor 
import logging
import time
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

template_result_queue = queue.Queue()
line_result_queue = queue.Queue()
serial_lock = threading.Lock()
exit_flag = threading.Event()
# 全局标志位：是否正在处理 LLM 对话
llm_active_flag = threading.Event()

lower = (50, 70, 60) 
upper = (255, 255, 255)

def filter_color_to_white(bgr_img,
                          lower_rgb=(0, 0, 0),   
                          upper_rgb=(255, 255, 255)):
    lower_bgr = lower_rgb[::-1]
    upper_bgr = upper_rgb[::-1]
    mask = cv2.inRange(bgr_img, lower_bgr, upper_bgr)  
    res  = bgr_img.copy()
    res[mask > 0] = (255, 255, 255)                  
    return res

# 相机线程函数
def camera_thread(shared_frame,track_frame):
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # cv2.nqamedWindow("Camera",cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Camera",640,480)

    while not exit_flag.is_set():
        ret, frame = cap.read()
        if ret:
            fra = filter_color_to_white(frame, lower, upper)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fra_gray = cv2.cvtColor(fra, cv2.COLOR_BGR2GRAY)
            # _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
            gray=cv2.GaussianBlur(gray,(3,3),0)
            # binary = binary[200:600, 200:1080]
            track_frame[0] = fra_gray
            shared_frame[0] = gray
            

    cap.release()
    # cv2.destroyAllWindows()

# 线程函数
def svm_classification_thread(shared_frame, model_path):
    classifier = TreeSVMClassifier(model_path)
    
    while not exit_flag.is_set():
        if shared_frame[0] is not None:
            result = classifier.predict(shared_frame[0])
            template_result_queue.put([{'name': result}])

# 巡线线程函数
def line_tracking_thread(track_frame):
    tracker = LineTracker()
    while not exit_flag.is_set():
        if track_frame[0] is not None:
            fra = track_frame[0]
            # fra=filter_color_to_white(fra,lower,upper)
            # fra = fra[0:480, 50:590]
            # gray = cv2.cvtColor(fra, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(fra, 120, 255, cv2.THRESH_BINARY)
            result = tracker.process_frame(binary)
            line_result_queue.put(result)


def llm_listener_thread(serial_port, llm):
    print("LLM监听线程已启动")
    while not exit_flag.is_set():
        if llm_active_flag.is_set():  # 暂停其他线程发送
            with serial_lock:
                print("in llm, please speak")
                try:
                    answer = llm.process_audio(duration=10)
                    if answer:
                        print(f"[LLM] 识别结果: {answer}")
                        serial_port.send_data(answer + ".")
                        print(f"[LLM] 已发送结果到串口: {answer}.")
                    else:
                        print("[LLM] 未检测到语音或处理失败")
                except Exception as e:
                    print(f"[LLM] 处理语音时出错: {str(e)}")
                
            time.sleep(5)
            llm_active_flag.clear()
            print("[LLM] 处理完成，重置标志")
        time.sleep(1)

def display_thread(shared_frame,serial_port):
    # cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Camera", 640, 480)
    """模板匹配名称映射"""
    matcher=TemplateMatcher(template_dir="mods",match_threshold=0.70)
    matcher.add_name_mapping("green1", "D2.")
    matcher.add_name_mapping("green2", "D2.")
    matcher.add_name_mapping("none1", "D1.")
    matcher.add_name_mapping("none2", "D1.")
    matcher.add_name_mapping("red1", "D1.")
    matcher.add_name_mapping("red2", "D1.")
    matcher.add_name_mapping("nopark", "Z.")
    matcher.add_name_mapping("nopa", "Z.")
    matcher.add_name_mapping("right", "RR.")
    matcher.add_name_mapping("back", "T.")

    current_state = "down"  
    # current_state = "up" 
    
    last_state_change_time = time.time()
    state_change_cooldown = 0.1 
    
    latest_line_result = None
    # latest_template_result = None

    while not exit_flag.is_set():
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            exit_flag.set()
            break

        match_state = "end"
        if shared_frame[0] is not None:
            # cv2.imshow("Camera", shared_frame[0])
            
            with serial_lock:
                received_data = serial_port.read_data()
                if received_data:
                    # if isinstance(received_data, str):
                    if "change1" in received_data.lower():
                        current_time = time.time()
                        if current_time - last_state_change_time > state_change_cooldown:
                            # current_state = "up" if current_state == "down" else "down"
                            current_state ="up"
                            last_state_change_time = current_time
                            print(f"set change: {current_state}")
                
                    elif "change2" in received_data.lower():
                        current_time = time.time()
                        if current_time - last_state_change_time > state_change_cooldown:
                            current_state = "down"
                            last_state_change_time = current_time
                            print(f"set change: {current_state}")
                    elif "start" in received_data.lower():
                        current_time = time.time()
                        if current_time - last_state_change_time > state_change_cooldown:
                            match_state = "start"
                            last_state_change_time = current_time
                            print(f"set match: {match_state}")
                    elif "change3" in received_data.lower():
                        current_time = time.time()
                        if current_time - last_state_change_time > state_change_cooldown:
                            llm_active_flag.set()  # 暂停其他线程发送
                            last_state_change_time = current_time
                            print(f"llm status: on")
                    # else:
                    #     continue
        
        while not line_result_queue.empty():
            latest_line_result = line_result_queue.get()
    


        with serial_lock:
            if not llm_active_flag.is_set():
                if current_state == "down" and latest_line_result is not None:
                    # latest_line_result +="."
                    # continue
                    serial_port.send_data(latest_line_result)
                elif current_state == "up":
                    if match_state=="start":
                        fram=shared_frame[0]
                        fram=fram[0:480, 0:640]
                        result=matcher.process_frame(fram)
                        if result:
                            time.sleep(0.2)
                            best_result = max(result, key=lambda x: x['confidence'])
                            serial_port.send_data(matcher.get_display_name(best_result['name']))
                            serial_port.send_data(matcher.get_display_name(best_result['name']))
                        else:
                            print("no find")
                            # serial_port.send_data("N.")
                            # continue
                    else:
                        continue
            
                        
        # key = cv2.waitKey(1)
        # if key & 0xFF == ord('q'):
        #     exit_flag.set()
        #     break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 创建串口实例
    serial_port = SerialPort(
        port='/dev/ttyAMA0',
        baudrate=115200,
        send_format='ascii',
        recv_format='ascii'
    )
    
    processor = LLMProcessor(
        api_key="e95feaa7f6ab4cdc807038f0b823a952.twjJFz6nrRg1GylJ",
        log_path="/home/hmkk/car_ws/log/dev.txt",
        corpus_dir="/home/hmkk/car_ws/new_corpus/corpus.txt",
        recorder_file="/home/hmkk/car_ws/MyRecorderAudio.wav"
    )
    # model_path = r'/home/hmkk/car_ws/tree_model.xml'
    # 共享帧变量
    shared_frame = [None]
    track_frame = [None]

    # 创建线程
    camera = threading.Thread(target=camera_thread, args=(shared_frame,track_frame))
    # svm_classification = threading.Thread(target=svm_classification_thread, args=(shared_frame,model_path))
    line_tracking = threading.Thread(target=line_tracking_thread, args=(track_frame,))
    display = threading.Thread(target=display_thread, args=(shared_frame,serial_port))
    llm_listener = threading.Thread(target=llm_listener_thread, args=(serial_port,processor))

    camera.daemon = True
    # svm_classification.daemon = True
    line_tracking.daemon = True
    llm_listener.daemon = True


    # 启动线程
    camera.start()
    # svm_classification.start()
    line_tracking.start()
    llm_listener.start()
    display.start()


    camera.join(timeout=1.0)
    # svm_classification.join(timeout=1.0)
    display.join(timeout=1.0)
    line_tracking.join(timeout=1.0)
    llm_listener.join(timeout=1.0)
