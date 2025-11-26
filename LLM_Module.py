import sys
import time
import os
import logging
from zhipuai import ZhipuAI
from threading import Thread, Event
import json
import traceback

class LLMProcessor:
    def __init__(self, api_key, log_path, corpus_dir, recorder_file):
        self.client = ZhipuAI(api_key=api_key.strip())
        self.LOG_PATH = log_path
        self.CORPUS_DIR = corpus_dir
        self.RECORDER_FILE = recorder_file
        self.messages = []
        self.llm_diag_btn_clicked = 0
        self.llm_source = 0
        self.last_response = ""
        self.current_text_input = ""
        self.microphone_card_no = "4c"
        self.microphone_device_no = "0"
        self._init_logger()
        self._init_microphone()
        self.event = Event()
        self.llm_thread = Thread(target=self._llm_task_loop, daemon=True)
        self.llm_thread.start()
        self.logger.info("LLMProcessor初始化完成")
        self._init_microphone(target_device_name="Device")

    def _init_logger(self):
        os.makedirs(os.path.dirname(self.LOG_PATH), exist_ok=True)
        
        logging.basicConfig(
            filename=self.LOG_PATH,
            format='%(asctime)s - %(name)s - %(levelname)s - %(module)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            # level=logging.DEBUG  # 使用DEBUG级别获取更多信息
        )
        self.logger = logging.getLogger()
        # 添加控制台输出
        console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        self.logger.info("日志系统初始化完成")

    def _init_microphone(self, target_device_name=None):
        try:
            # 获取录音设备列表
            result = os.popen("arecord -l").read()
            lines = result.split('\n')
            found = False
            
            for line in lines:
                # 查找包含设备信息的行
                if "card" in line and "device" in line:
                    # 提取卡片信息部分（如 "card 0: PCH [HDA Intel PCH]"）
                    card_part = line.split(',')[0].strip()
                    # 提取设备信息部分（如 "device 0: ALC256 Analog [ALC256 Analog]"）
                    device_part = [p.strip() for p in line.split(',') if 'device' in p][0]
                    
                    # 解析卡片编号
                    card_no = card_part.split(':')[0].split()[-1]
                    # 解析设备编号
                    device_no = device_part.split(':')[0].split()[-1]
                    
                    # 提取设备名称（通常在方括号中）
                    # 卡片名称（如 "HDA Intel PCH"）
                    card_name = card_part.split('[')[-1].split(']')[0] if '[' in card_part else ''
                    # 设备名称（如 "ALC256 Analog"）
                    device_name = device_part.split('[')[-1].split(']')[0] if '[' in device_part else ''
                    full_device_name = f"{card_name} {device_name}".strip()
                    
                    # 如果指定了目标名称，则匹配名称；否则取第一个设备
                    if not target_device_name:
                        # 未指定目标名称，使用第一个找到的设备
                        self.microphone_card_no = card_no
                        self.microphone_device_no = device_no
                        self.logger.info(
                            f"使用第一个找到的麦克风设备: {full_device_name}, "
                            f"card: {card_no}, device: {device_no}"
                        )
                        return True
                    else:
                        # 匹配目标设备名称（不区分大小写）
                        if target_device_name.lower() in full_device_name.lower():
                            self.microphone_card_no = card_no
                            self.microphone_device_no = device_no
                            self.logger.info(
                                f"找到匹配的麦克风设备: {full_device_name}, "
                                f"card: {card_no}, device: {device_no}"
                            )
                            found = True
                            break
            
            if target_device_name and not found:
                self.logger.error(f"未找到名称包含 '{target_device_name}' 的麦克风设备")
                return False
                
            # 如果没有找到任何设备
            if not found and not target_device_name:
                self.logger.error("未检测到任何麦克风设备")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"麦克风初始化失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def _format_date_output(self, info_str, elapse_time=""):
        ct = time.time()
        local_time = time.localtime(ct)
        formatted_datetime = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
        if elapse_time == "":
            return f'{formatted_datetime}: {info_str}'
        else:
            return f'{formatted_datetime}: {info_str} ({elapse_time*1000:.2f}ms)'

    def _llm_call(self, messages):
        try:
            # self.logger.debug(f"发送给大模型的消息: {json.dumps(messages, ensure_ascii=False, indent=2)}")
            
            start_time = time.time()
            response = self.client.chat.completions.create(
                model="GLM-4-Flash-250414",
                messages=messages
            )
            end_time = time.time()
            
            if not response.choices:
                raise Exception("大模型返回空响应")
            
            answer = response.choices[0].message.content.strip()
            elapsed = end_time - start_time
            
            self.logger.info(f"大模型调用成功, 耗时: {elapsed:.2f}s")
            # self.logger.debug(f"大模型原始响应: {answer}")
            
            return answer, elapsed
        except Exception as e:
            self.logger.error(f"大模型调用失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            return f"调用失败: {str(e)}", time.time() - start_time

    def _recorder(self, duration=5):
        try:
            os.makedirs(os.path.dirname(self.RECORDER_FILE), exist_ok=True)
            
            cmd_str = (f"arecord -D hw:{self.microphone_card_no},{self.microphone_device_no} "
                       f"-f cd -d {duration} -c 1 {self.RECORDER_FILE}")
            
            self.logger.info(f"执行录音命令: {cmd_str}")
            return_code = os.system(cmd_str)
            
            if return_code != 0:
                raise Exception(f"录音命令失败, 返回码: {return_code}")
            
            if not os.path.exists(self.RECORDER_FILE):
                raise Exception("录音文件未创建")
                
            if os.path.getsize(self.RECORDER_FILE) == 0:
                raise Exception("录音文件为空")
                
            self.logger.info(f"录音成功, 文件: {self.RECORDER_FILE}, 大小: {os.path.getsize(self.RECORDER_FILE)}字节")
            return True
        except Exception as e:
            self.logger.error(f"录音失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def _recorder_llm_call(self):
        try:
            self.logger.info("开始语音识别...")
            
            if not os.path.exists(self.RECORDER_FILE):
                raise Exception(f"录音文件不存在: {self.RECORDER_FILE}")
                
            if os.path.getsize(self.RECORDER_FILE) == 0:
                raise Exception(f"录音文件为空: {self.RECORDER_FILE}")
            
            with open(self.RECORDER_FILE, "rb") as audio_file:
                start_time = time.time()
                
                # 调用语音识别API
                response = self.client.audio.transcriptions.create(
                    model="glm-asr",
                    file=audio_file,
                    stream=False
                )
                
                end_time = time.time()
                elapsed = end_time - start_time
                
                if not response.text.strip():
                    raise Exception("语音识别结果为空")
                
                text = response.text.strip()
                self.logger.info(f"语音识别成功: '{text}', 耗时: {elapsed:.2f}s")
                return text, elapsed, True
        except Exception as e:
            self.logger.error(f"语音识别失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            return f"识别失败: {str(e)}", time.time() - start_time, False

    def _load_corpus(self, index):
        """加载指定索引的语料文件"""
        try:
            base_name = self.CORPUS_DIR.rstrip('.txt')
            corpus_file = f"{base_name}{index}.txt"
            
            self.logger.info(f"加载语料文件: {corpus_file}")
            
            if not os.path.exists(corpus_file):
                raise FileNotFoundError(f"语料文件不存在: {corpus_file}")
            
            with open(corpus_file, "r", encoding='utf-8') as f:
                content = f.read().strip()
                
            # self.logger.debug(f"语料 {index} 内容: {content[:100]}...")  # 只记录前100字符
            return content
        except Exception as e:
            self.logger.error(f"加载语料失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            return f"语料加载错误: {str(e)}"

    def _llm_init(self):
        """大模型初始化流程，加载3个语料文件"""
        try:
            self.logger.info("开始大模型初始化...")
            self.messages = []
            for i in range(1, 4):
                content = self._load_corpus(i)
                self.messages.append({"role": "user", "content": content})
                answer, t = self._llm_call(self.messages)
                if i == 3:
                    if not answer.endswith("bee(0.5)"):
                        answer += ",bee(0.5)"
                        self.logger.info("添加bee(0.5)到第三条回复")

                self.messages.append({"role": "assistant", "content": answer})
                
                self.logger.info(f"语料 {i} 处理完成")
            
            self.logger.info("大模型初始化完成")
            return True
        except Exception as e:
            self.logger.error(f"大模型初始化失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def _llm_task_loop(self):
        """大模型任务主循环"""
        try:
            if not self._llm_init():
                self.logger.error("大模型初始化失败，无法继续")
                return
                
            self.logger.info("进入大模型任务主循环")
            
            while True:
                if self.llm_diag_btn_clicked:
                    try:
                        self.logger.info("处理新请求...")
                        self.llm_diag_btn_clicked = 0
                        
                        if self.llm_source == 0: 
                            info_str = self.current_text_input
                            self.logger.info(f"文本输入: '{info_str}'")
                        else: 
                            self.logger.info("处理语音输入...")
                            
                            if not self._recorder():
                                self.last_response = "录音失败"
                                self.event.set()
                                continue
                            
                            text, _, success = self._recorder_llm_call()
                            if not success:
                                self.last_response = text 
                                self.event.set()
                                continue
                                
                            info_str = text
                            self.logger.info(f"语音识别结果: '{info_str}'")
                        
                        if not info_str or not info_str.strip():
                            self.logger.warning("输入内容为空")
                            self.last_response = "没有输入内容，请重新输入"
                            self.event.set()
                            continue
                        
                        self.messages.append({"role": "user", "content": info_str})
                        
                        answer, t = self._llm_call(self.messages)
                        
                        if not answer or not answer.strip():
                            self.logger.warning("大模型返回空响应，尝试重新调用...")
                        
                            answer, t = self._llm_call(self.messages)
                            
                            if not answer or not answer.strip():
                                self.logger.error("大模型再次返回空响应")
                                answer = "NULL"  

                        self.messages.append({"role": "assistant", "content": answer})
                        
                        self.logger.info(f"大模型响应: '{answer}'")
                        self.last_response = answer
                    except Exception as e:
                        self.logger.error(f"处理请求时出错: {str(e)}")
                        self.logger.error(traceback.format_exc())
                        self.last_response = f"处理错误: {str(e)}"
                    finally:
                        self.event.set()
                
                time.sleep(0.1)
        except Exception as e:
            self.logger.error(f"大模型任务循环崩溃: {str(e)}")
            self.logger.error(traceback.format_exc())

    def process_text(self, text):
        """处理文本输入"""
        self.logger.info(f"处理文本输入: '{text}'")
        self.current_text_input = text
        self.llm_source = 0
        self.llm_diag_btn_clicked = 1
        self.event.clear()
        self.event.wait()
        return self.last_response

    def process_audio(self, duration=5):
        """处理语音输入"""
        self.logger.info(f"处理语音输入, 时长: {duration}秒")
        self.llm_source = 1
        self.llm_diag_btn_clicked = 1
        self.event.clear()
        self.event.wait() 
        return self.last_response


if __name__ == "__main__":
    # 配置参数 - 根据您的环境调整
    API_KEY = "e95feaa7f6ab4cdc807038f0b823a952.twjJFz6nrRg1GylJ"
    LOG_PATH = "/home/hmkk/car_ws/log/dev.txt"
    CORPUS_DIR = "/home/hmkk/car_ws/new_corpus/corpus.txt"  
    RECORDER_FILE = "/home/hmkk/car_ws/MyRecorderAudio.wav" 
    
    print("初始化LLM处理器...")
    processor = LLMProcessor(
        api_key=API_KEY,
        log_path=LOG_PATH,
        corpus_dir=CORPUS_DIR,
        recorder_file=RECORDER_FILE
    )
    
    # 等待初始化完成
    time.sleep(5)
    print("初始化完成，开始测试...")
    
    # # 测试1: 文本处理
    # print("\n测试文本处理...")
    # text_response = processor.process_text("低速前进五秒")
    # print(f"文本响应: {text_response}")
    
    # 测试2: 语音处理
    print("\n测试语音处理...")
    print("请说话（5秒录音）...")
    audio_response = processor.process_audio(duration=5)
    print(f"语音响应: {audio_response}")
    
    # # 测试3: 显示当前消息历史
    # print("\n当前消息历史:")
    # for i, msg in enumerate(processor.messages):
    #     role = msg["role"]
    #     content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
    #     print(f"{i+1}. {role.upper()}: {content}")