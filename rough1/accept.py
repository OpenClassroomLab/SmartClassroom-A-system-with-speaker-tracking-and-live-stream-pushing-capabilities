import time
import sounddevice as sd
import numpy as np

# 变量来记录检测到声音的时间戳
sound_detected_time = None

# 声音检测回调函数
def detect_sound(indata, frames, time, status):
    global sound_detected_time
    if np.max(indata) > 0.1: # 假设0.1为检测声音的阈值
        sound_detected_time = time.inputBufferAdcTime
        print(f"收到声音时间戳：{sound_detected_time:.6f}")
        sd.stop()

# 打开音频流并监听麦克风输入
stream = sd.InputStream(callback=detect_sound)
with stream:
    print("等待声音输入...")
    while sound_detected_time is None:
        time.sleep(0.1)

# 计算延迟
if sound_detected_time:
    # 使用实际的发声时间戳
    start_time = float(input("请输入发声时间戳："))
    delay = sound_detected_time - start_time
    delay_ms = delay * 1000
    print(f"延迟：{delay_ms:.2f} 毫秒")

