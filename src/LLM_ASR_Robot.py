import sys
import time
import os

from PyQt5.QtCore import QLibraryInfo
from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
#import whisper

import json
import logging

# paddlespeech
#from paddlespeech.cli.asr.infer import ASRExecutor
#asr = ASRExecutor()

# whisper，https://blog.csdn.net/huiguo_/article/details/135080485
# tiny、base、small、medium、large
#model = whisper.load_model("tiny")

# export DISPLAY=:10.0
# 用于从麦克风采集语音
import pyaudio
import wave

import serial
import time

from zhipuai import ZhipuAI
client = ZhipuAI(api_key="e95feaa7f6ab4cdc807038f0b823a952.twjJFz6nrRg1GylJ") # 更换为自己的api-key

LOG_PATH = "/home/hmkk/car_ws/log/dev.txt"
CORPUS_DIR = "/home/hmkk/car_ws/corpus/corpus.txt"
RECORDER_FILE = "/home/hmkk/car_ws/MyRecorderAudio.mp3"
RECORDER_FILE2 = "/home/hmkk/car_ws/test.mp3"

logging.basicConfig(filename=LOG_PATH,
                    format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S ',
                    level=logging.INFO)
logger = logging.getLogger()

# ser = serial.Serial("COM3",115200)
ser = serial.Serial("/dev/ttyAMA0",115200)

llm_diag_btn_clicked = 0        # 是否按下“大模型文本对话”按钮
llm_diag_wave_btn_clicked = 0   # 是否按下“大模型语音对话”按钮
llm_source = 0                  # 大模型对话的问题来源，0-来自文本框，1-来自音频
llm_wave_question = ""          # 大模型问题

# 加载.ui文件为Python对象
class MyWidget(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWidget, self).__init__()
        # ui文件的路径
        #self.ui = uic.loadUi('./LLM_ASR_Robot/test.ui', self)
        self.ui = uic.loadUi('/home/hmkk/car_ws/test.ui', self)
        self.show()

        ### 手动控制小车运动的按钮
        # 前进、后退、左转、右转按钮
        self.Forward_btn = self.ui.Forward_btn
        self.Backward_btn = self.ui.Backward_btn
        self.TurnLeft_btn = self.ui.TurnLeft_btn
        self.TurnRight_btn = self.ui.TurnRight_btn
        self.Stop_btn = self.ui.Stop_btn
        self.Bee_btn = self.ui.Bee_btn
        # 绑定上述按钮的点击事件
        self.Forward_btn.clicked.connect(self.on_forward_btn_clicked)
        self.Backward_btn.clicked.connect(self.on_backward_btn_clicked)
        self.TurnLeft_btn.clicked.connect(self.on_turnleft_btn_clicked)
        self.TurnRight_btn.clicked.connect(self.on_turnright_btn_clicked)
        self.Stop_btn.clicked.connect(self.on_stop_btn_clicked)
        self.Bee_btn.clicked.connect(self.on_bee_btn_clicked)
        
        layout = QVBoxLayout()

        # 设置移动距离的滑动条
        self.MoveTimer_Slider = self.ui.MoveTimer_Slider #QSlider(Qt.Horizontal) # 生成水平方向滚动条
        self.MoveTimer_lineEdit = self.ui.MoveTimer_lineEdit
        self.MoveTimer_Slider.setRange(1, 5) # 设置范围
        self.MoveTimer_Slider.setSingleStep(1) # 设置步长
        self.MoveTimer_Slider.setValue(2)     # 设置当前值
        self.MoveTimer_Slider.setTickPosition(QSlider.TicksBelow)  # 设置刻度位置，在下方
        self.MoveTimer_Slider.setTickInterval(1)   # 设置刻度间隔
        self.MoveTimer_Slider.valueChanged.connect(self.move_timer_slider_changed)
        move_timer = self.MoveTimer_Slider.value()    # 获取运动距离
        self.MoveTimer_lineEdit.setText(str(move_timer))  # 写入文本框
        
        # 设置移动速度的滑动条
        self.MoveSpeed_Slider = self.ui.MoveSpeed_Slider #QSlider(Qt.Horizontal) # 生成水平方向滚动条
        self.MoveSpeed_lineEdit = self.ui.MoveSpeed_lineEdit
        self.MoveSpeed_Slider.setRange(1, 10)   # 设置速度范围（数值要除10），单位cm/s
        self.MoveSpeed_Slider.setSingleStep(1) # 设置步长
        self.MoveSpeed_Slider.setValue(5)      # 设置当前值
        self.MoveSpeed_Slider.setTickPosition(QSlider.TicksBelow)  # 设置刻度位置，在下方
        self.MoveSpeed_Slider.setTickInterval(1)   # 设置刻度间隔
        self.MoveSpeed_Slider.valueChanged.connect(self.move_speed_slider_changed)
        move_speed = self.MoveSpeed_Slider.value()    # 获取运动速度
        self.MoveSpeed_lineEdit.setText(str(move_speed))  # 写入文本框
        #self.setLayout(layout)

        ### 信息输入输出窗口和按钮
        # 信息窗口-进行大模型对话
        self.LLM_Info_Editor = self.ui.LLM_Info_Editor
        # 添加文本并自动滚动到最底部
        self.LLM_Info_Editor.clear()
        self.LLM_Info_Editor.append("请在这里输入文字与控制小车动作的大模型对话。例如：高速前进5.5秒")
        
        # 信息窗口-显示日志信息
        self.Info_Editor = self.ui.Info_Editor
        # 添加文本并自动滚动到最底部
        self.Info_Editor.clear()
        
        # 按下LLM_Diag_btn按钮，可以读取LLM_Info_Editor中的信息，并调用大模型获取对话结果
        self.LLM_Diag_btn = self.ui.LLM_Diag_btn
        self.LLM_Diag_btn.clicked.connect(self.on_llm_diag_btn_clicked)
    
        # 按下LLM_Diag_Wave_btn按钮，可以采集麦克风中的语音信息，并调用大模型获取对话结果
        self.LLM_ASR_Diag_btn = self.ui.LLM_ASR_Diag_btn
        self.LLM_ASR_Diag_btn.clicked.connect(self.on_llm_asr_diag_btn_clicked)
        
        # 信息窗口-进行大模型对话
        self.Info_Editor = self.ui.Info_Editor
        # 添加文本并自动滚动到最底部
        self.Info_Editor.clear()
        
        
        # 测试语音识别大模型（实际上是因为，通过测试发现，在下面大模型线程开启后，第一次调用语音大模型非常慢，因此在开启大模型线程之前先测试一次，提高速度）
        #self.Recorder_LLM_Call(RECORDER_FILE2)
        
        # 启动大模型线程
        self.mThread_LLM_task = LLM_Task_Thread(self.LLM_Invoke)
        self.mThread_LLM_task.start()
        
        # 大模型初始化
        #self.LLM_Init()

        # 确认麦克风的端口
        #return_value = os.system("arecord -l")
        # 捕获命令输出
        global microphone_card_no, microphone_device_no
        microphone_str = os.popen("arecord -l").read()
        index1 = microphone_str.find("card")
        index2 = microphone_str.find("device")
        microphone_card_no = microphone_str[index1+5]
        microphone_device_no = microphone_str[index2+7]
        info_str = "麦克风初始化完成，card %s, device %s" % (microphone_card_no, microphone_device_no)
        info_str = self.FormatDate_Output(info_str, "")
        self.Info_Editor.append(info_str)  # 写入文本框

        # 串口初始化
        if ser.isOpen():
            info_str = "串口已打开"
            info_str = self.FormatDate_Output(info_str, "")
            self.Info_Editor.append(info_str)  # 写入文本框
        else:
            info_str = "串口未打开"
            info_str = self.FormatDate_Output(info_str, "")
            self.Info_Editor.append(info_str)  # 写入文本框

        
        logger.info("GUI界面初始化完成GUI initialization Completed...")
        

    # 格式化输出，将要显示的信息前面加上当前时间戳
    def FormatDate_Output(self, info_str, elapse_time):
        ct = time.time()
        local_time = time.localtime(ct)
        # 格式化当前日期和时间
        formatted_datetime = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
        if elapse_time=="":
            info_str = '%s: %s' % (formatted_datetime, info_str)
        else:
            info_str = '%s: %s (%.2fms)' % (formatted_datetime, info_str, elapse_time*1000)
        return info_str
    
    def SerialWrite(self, info_str):
        try:
            # 发送数据
            ser.write(info_str.encode("utf-8"))
            # 等待回复
            time.sleep(1)
            # 读取数据
            response = ser.read_all()
            print(f"从串口接收到的数据: {response}")
        except serial.SerialTimeoutException as e:
            print(f"串口通信超时: {e}")
        except serial.SerialException as e:
            print(f"串口通信出错: {e}")
        #finally:
            #ser.close()
            #print("串口已关闭")

    # 按钮点击事件的处理函数
    def on_forward_btn_clicked(self):
        logger.info("按下前进按钮...")
        move_timer = self.MoveTimer_Slider.value()    # 获取运动距离
        move_speed = self.MoveSpeed_Slider.value()    # 获取运动速度
        fcn_str = 'forward_ctr(%d, %d)' % (move_timer, move_speed)
        info = '[Mannual]: %s' % fcn_str
        info = self.FormatDate_Output(info, "")
        self.Info_Editor.append(info)

        # 串口发送
        self.SerialWrite(fcn_str)
        
        self.Info_Editor.moveCursor(self.Info_Editor.textCursor().End)  #文本框显示到底部
        time.sleep(0.2)
        
    def on_backward_btn_clicked(self):
        logger.info("按下后退按钮...")
        move_timer = self.MoveTimer_Slider.value()    # 获取运动距离
        move_speed = self.MoveSpeed_Slider.value()    # 获取运动速度
        fcn_str = 'backward_ctr(%d, %d)' % (move_timer, move_speed)
        info = '[Mannual]: %s' % fcn_str
        info = self.FormatDate_Output(info, "")
        self.Info_Editor.append(info)
        
        # 串口发送
        self.SerialWrite(fcn_str)

        self.Info_Editor.moveCursor(self.Info_Editor.textCursor().End)  #文本框显示到底部
        time.sleep(0.2)

    def on_turnleft_btn_clicked(self):
        logger.info("按下左转按钮...")
        move_timer = self.MoveTimer_Slider.value()    # 获取运动距离
        move_speed = self.MoveSpeed_Slider.value()    # 获取运动速度
        fcn_str = 'left_ctr(%d, %d)' % (move_timer, move_speed)
        info = '[Mannual]: %s' % fcn_str
        info = self.FormatDate_Output(info, "")
        self.Info_Editor.append(info)
        
        # 串口发送
        self.SerialWrite(fcn_str)

        self.Info_Editor.moveCursor(self.Info_Editor.textCursor().End)  #文本框显示到底部
        time.sleep(0.2)

    def on_turnright_btn_clicked(self):
        logger.info("按下右转按钮...")
        move_timer = self.MoveTimer_Slider.value()    # 获取运动距离
        move_speed = self.MoveSpeed_Slider.value()    # 获取运动速度
        fcn_str = 'right_ctr(%d, %d)' % (move_timer, move_speed)
        info = '[Mannual]: %s' % fcn_str
        info = self.FormatDate_Output(info, "")
        self.Info_Editor.append(info)
        
        # 串口发送
        self.SerialWrite(fcn_str)

        self.Info_Editor.moveCursor(self.Info_Editor.textCursor().End)  #文本框显示到底部
        time.sleep(0.2)

    def on_stop_btn_clicked(self):
        logger.info("按下停车按钮...")
        fcn_str = 'stop()'
        info = '[Mannual]: %s' % fcn_str
        info = self.FormatDate_Output(info, "")
        self.Info_Editor.append(info)
        
        # 串口发送
        self.SerialWrite(fcn_str)

        self.Info_Editor.moveCursor(self.Info_Editor.textCursor().End)  #文本框显示到底部
        time.sleep(0.2)

    def on_bee_btn_clicked(self):
        logger.info("按下蜂鸣器按钮...")
        move_timer = self.MoveTimer_Slider.value()    # 获取运动距离
        fcn_str = 'bee(%d)' % (move_timer)
        info = '[Mannual]: %s' % fcn_str
        info = self.FormatDate_Output(info, "")
        self.Info_Editor.append(info)
        
        # 串口发送
        self.SerialWrite(fcn_str)

        self.Info_Editor.moveCursor(self.Info_Editor.textCursor().End)  #文本框显示到底部
        time.sleep(0.2)
        
    def move_timer_slider_changed(self):
        #print('修改当前值为：' + str(self.MoveTimer_Slider.value()))
        move_timer = self.MoveTimer_Slider.value()    # 获取运动距离
        self.MoveTimer_lineEdit.setText(str(move_timer))  # 写入文本框
        
    def move_speed_slider_changed(self):
        move_speed = self.MoveSpeed_Slider.value()    # 获取运动速度
        self.MoveSpeed_lineEdit.setText(str(move_speed))  # 写入文本框
        
    def on_llm_diag_btn_clicked(self):
        global llm_diag_btn_clicked
        global llm_source
        llm_source = 0 # 根据文本框输入进行大模型对话
        llm_diag_btn_clicked = 1
        
    def on_llm_asr_diag_btn_clicked(self):
        logger.info("等待语音输入...")
        self.Info_Editor.append("请在5秒内完成语音输入...")  # 写入文本框
        #audio_file = "./LLM_ASR_Robot/录音.mp3"
        # 调用自带麦克风录音
        # self.Recorder(audio_file)
        # 调用树莓派上的usb麦克风录音
        global microphone_card_no, microphone_device_no
        cmd_str = "arecord -D hw:"+ microphone_card_no +","+ microphone_device_no +" -f cd -d 5 -c 1 " + RECORDER_FILE  # -D hw:0,0: 指定录音设备。-d 5: 指定录音时长为5秒。-f cd: 指定录音格式为CD质量。-r 44100: 指定采样率为44100Hz。-t wav: 指定输出文件类型为WAV
        return_value = os.system(cmd_str)
        if return_value != 0: # 如果代码执行失败，在窗口中显示并退出录音，否则做下面的大模型语音识别
            info_str = "录音失败，%s，请重新录制..." % os.popen(cmd_str).read()
            self.Info_Editor.append(info_str)  # 写入文本框
            return
        else:
            logger.info("完成语音输入...")

        timer1 = time.time()
        #result = model.transcribe(audio_file)
        
        with open(RECORDER_FILE, "rb") as audio_file1:
            transcriptResponse = client.audio.transcriptions.create(
                model="glm-asr",
                file=audio_file1,
                stream=False
            )
        timer2 = time.time()
        info = '[Microphone]: %s, %d ms' % (transcriptResponse.text, (timer2-timer1)*1000)
        info = self.FormatDate_Output(info, "")
        self.Info_Editor.append(info)  # 写入文本框
        self.LLM_Info_Editor.setText(transcriptResponse.text)  # 写入文本框，用于提供给大模型对话
        
        logger.info("录音内容为：" + info)

        global llm_diag_btn_clicked
        global llm_source
        llm_source = 1 # 根据语音输入进行大模型对话
        llm_diag_btn_clicked = 1
        

    def LLM_Init(self):
        print("开始大模型初始化")
        # 大模型初始化
        messages = []
        for i in range(1, 4):
            with open(CORPUS_DIR.split(".txt")[0] + str(i) + ".txt", "r", encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines()]
                content = "".join(lines)
                messages.append({"role": "user", "content": content})
                data_str = json.dumps({
                    "messages": messages,
                    "stream": False
                })
            
            answer, t = self.LLM_Call(messages)
            info = '[LLM_Answer]: %s' % answer
            info = self.FormatDate_Output(info, t)
            # 将大模型输出加入messages，形成上下文
            messages.append({"role": "assistant", "content": answer})
            # 将大模型输出显示到editor
            self.Info_Editor.append(info)
        
        self.Info_Editor.moveCursor(self.Info_Editor.textCursor().End)  #文本框显示到底部
        time.sleep(0.2)
        
        # 完成大模型初始化logger
        # print("完成大模型初始化")
        logger.info("大模型初始化完成LLM Initialization Completed...")
    
    
    # 大模型调用
    def LLM_Invoke(self):
        #print("开始大模型初始化")
        logger.info("开始大模型初始化LLM Initialization...")
        # 大模型初始化
        messages = []
        for i in range(1, 4):
            with open(CORPUS_DIR.split(".txt")[0] + str(i) + ".txt", "r", encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines()]
                content = "".join(lines)
                messages.append({"role": "user", "content": content})
                data_str = json.dumps({
                    "messages": messages,
                    "stream": False
                })
            
            answer, t = self.LLM_Call(messages)
            info = '[LLM_Answer]: %s' % answer
            info = self.FormatDate_Output(info, t)
            # 将大模型输出加入messages，形成上下文
            messages.append({"role": "assistant", "content": answer})
            # 将大模型输出显示到editor
            self.Info_Editor.append(info)
        
        self.Info_Editor.moveCursor(self.Info_Editor.textCursor().End)  #文本框显示到底部
        time.sleep(0.2)
        
        # 完成大模型初始化logger
        #print("完成大模型初始化")
        logger.info("完成大模型初始化LLM Initialization Completed...")
        
        # 大模型连续调用
        while True:
            global llm_diag_btn_clicked
            global llm_source
            if llm_diag_btn_clicked==1:
                print(llm_diag_btn_clicked)
                llm_diag_btn_clicked = 0
                
                # (1) 读取LLM_Info_Editor内容
                if llm_source == 0:
                    info_str = self.LLM_Info_Editor.toPlainText()
                    logger.info("llm_source: LLM_Info_Editor")
                    #print("llm_source: LLM_Info_Editor")
                    
                    # (2) 将问题显示到日志窗口
                    info = '[LLM_Question]: %s' % info_str
                    info = self.FormatDate_Output(info, "")
                    self.Info_Editor.append(info)

                elif llm_source == 1:
                    info_str = self.LLM_Info_Editor.toPlainText()
                    logger.info("llm_source: recorder")
                    #print("llm_source: recorder")
                
                # 如果没有任何输入，则不做任何响应，并在日志窗口输出提示
                if len(info_str) == 0:
                    info = self.FormatDate_Output("没有输入问题，请重新输入", "")
                    # 将大模型输出加入messages，形成上下文
                    messages.append({"role": "assistant", "content": info})
                    # 将大模型输出显示到editor
                    self.Info_Editor.append(info)
                    return
                else:
                    # (3) 调用大模型并将大模型输出显示到日志窗口
                    messages.append({"role": "user", "content": info_str})
                    #print(messages)
                    logger.info(messages)
                    
                    answer, t = self.LLM_Call(messages)
                    info = '[LLM_Answer]: %s' % answer
                    info = self.FormatDate_Output(info, t)
                    
                    # 将大模型输出加入messages，形成上下文
                    messages.append({"role": "assistant", "content": answer})
                    # 将大模型输出显示到editor
                    self.Info_Editor.append(info)
        
                    self.Info_Editor.moveCursor(self.Info_Editor.textCursor().End)  #文本框显示到底部
                    time.sleep(0.2)
                    
                # (4) 最后清空 LLM_Info_Editor
                #self.LLM_Info_Editor.clear()
                #self.LLM_Info_Editor.setText("")
            
    
    # 调用大模型（同步方式）
    def LLM_Call(self, messages):
        start_time = time.time()
        #（1）同步调用
        response = client.chat.completions.create(
            model = "GLM-4-Flash-250414",#"glm-4-plus",  # 请填写您要调用的模型名称
            messages = messages,
        )
        end_time = time.time()
        return response.choices[0].message.content, end_time-start_time
    
    def Recorder(self, file):
        # 设置录音参数
        CHUNK = 1024  # 每个缓冲区的帧数
        FORMAT = pyaudio.paInt16 # 采样位数
        CHANNELS = 1  # 声道数，目前智谱的大模型似乎只支持单声道
        RATE = 44100  # 采样率
        RECORD_SECONDS = 5 # 录音时长
        # 创建音频流
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        # 开始录音
        print("开始录音")
        frames = []
        for i in range(0, int(RATE/CHUNK*RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("录音结束")

        # 关闭音频流
        stream.stop_stream()
        stream.close()
        p.terminate()

        # 保存音频文件
        wf = wave.open(file, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()


    # 调用语音识别大模型
    def Recorder_LLM_Call(self, recorder_file):
        
        '''
        start_time = time.time()
        result = model.transcribe(recorder_file)
        end_time = time.time()
        return result['text'], end_time-start_time
        '''
        

        start_time = time.time()
        #result = model.transcribe(recorder_file)
        
        print(recorder_file)
        with open(recorder_file, "rb") as audio_file:
            transcriptResponse = client.audio.transcriptions.create(
                model="glm-asr",
                file=audio_file,
                stream=False
            )
            print(transcriptResponse.text)
                
        end_time = time.time()
        return transcriptResponse.text, end_time-start_time
        '''
        start_time = time.time()
        result = asr(audio_file=recorder_file, force_yes=True)
        end_time = time.time()
        return result, end_time-start_time
        '''

class LLM_Task_Thread(QThread):
    def __init__(self, run):
        super(LLM_Task_Thread, self).__init__()
        self.runfun = run

    def run(self):
        self.runfun()

if __name__=='__main__':

    logger.info("主程序初始化Main Program initialization...")

    app = QApplication(sys.argv)
    w = MyWidget()
    w.ui.show()
        
    app.exec()
    sys.exit(app.exec_())