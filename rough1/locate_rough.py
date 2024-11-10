import socket
import time
import threading
from multiprocessing import Process, Pipe
from datetime import datetime, timedelta
from onvif_hik import Onvif_hik
from datetime import datetime, timedelta
import threading
from loguru import logger



# 连接摄像头

camera_connect_time1 = time.time()
while True:
    camera30 = Onvif_hik(ip='...', username="...", password="...")
    if camera30.content_cam:
        break
while True:
    camera31 = Onvif_hik(ip='...', username="...", password="...")
    if camera31.content_cam:
        break
while True:
    camera32 = Onvif_hik(ip='...', username="...", password="...")
    if camera32.content_cam:
        break
while True:
    camera33 = Onvif_hik(ip='...', username="...", password="...")
    if camera33.content_cam:
        break

camera_connect_time2 = time.time()

logger.info(f'连接摄像头完成，总用时（s)：{camera_connect_time2-camera_connect_time1}')

                                    
# 绿色麦克风40
mic40 = [0, 0, 0, 15, 13, 11, 16, 14, 12]
# 蓝色麦克风42
mic42 = [0, 12, 18, 19, 21, 20, 17, 22, 23]

# number of cameras
MAX = 4

# Priority dictionary
priority_dict = {
    11: [31, 30, 32, 33],
    12: [31, 30, 32, 33],
    13: [30, 31, 33, 32],
    14: [31, 30, 32, 33],
    15: [30, 31, 33, 32],
    16: [30, 31, 33, 32],
    17: [31, 30, 32, 33],
    18: [30, 31, 32, 33],
    19: [31, 30, 33, 32],
    20: [31, 30, 32, 33],
    21: [30, 31, 33, 32],
    22: [33, 32, 31, 30],
    23: [32, 33, 30, 31]
}

# Camera work status array, 0 means available and 1 means occupied
ifwork = [0] * MAX
camera_occupation = [0] * MAX
preset_occupation = [0] * 24

last_activity_times = [None] * MAX  # 索引0对应摄像头30，索引4对应摄像头34

# Lock
camera_data_lock = threading.Lock()
last_activity_lock = threading.Lock()

# Initialize pipes
conn1_a, conn1_b = Pipe()  # Pipe for microphone 1
conn2_a, conn2_b = Pipe()  # Pipe for microphone 2


def choose_camera(preset):
    # 输出当前占用的摄像头和对应的预置点
    for i in range(MAX):
        if ifwork[i]:
            logger.info(f"摄像头{30+i}：预置点{camera_occupation[i]}")
    # 检查预置点是否已被拍摄
    if preset_occupation[preset] != 0:
        # 获取正在使用该预置点的摄像头编号
        active_camera = preset_occupation[preset]
        # 更新摄像头的活动时间
        with last_activity_lock:
            last_activity_times[active_camera - 30] = datetime.now()
        logger.info(f"预置点 {preset} 已经有摄像头 {active_camera} 拍摄。")
        return -1  # 返回 -1 表示预置点已被占用
        
        # 寻找可用的摄像头
    for priority_camera in priority_dict[preset]:
        index = priority_camera - 30
        if not ifwork[index]:  # 检查摄像头是否可用
            with camera_data_lock:
                ifwork[index] = 1
                preset_occupation[preset] = priority_camera
                camera_occupation[index] = preset
            with last_activity_lock:
                last_activity_times[index] = datetime.now()  # 初始化活动时间
            logger.info(f"Camera {priority_camera} is now in use for preset {preset}.")
            return priority_camera  # 返回实际摄像头编号
    return None  # 如果没有可用的摄像头，则返回 None

def use_camera(preset):
    # Choose camera
    chosen_camera = choose_camera(preset)

    #预置点已被拍摄
    if chosen_camera == -1 :
        return 

    if chosen_camera is not None:

        # logger.info('摄像头运动开始')
        if chosen_camera == 30:
            camera30.goto_preset(preset)
        if chosen_camera == 31:
            camera31.goto_preset(preset)
        if chosen_camera == 32:
            camera32.goto_preset(preset)
        if chosen_camera == 33:
            camera33.goto_preset(preset)
        # logger.info('摄像头运动完毕')


        logger.info(f'Camera {chosen_camera} is now in use for preset {preset}.') 
        
    else:
        logger.info(f"No available cameras for preset {preset}.")
    

def release_camera():
    current_time = datetime.now()
    # logger.info(f'Checking for camera release at {current_time}')
    for index in range(MAX):
        last_time = last_activity_times[index]
        logger.info(f"Camera {index + 30} last activity time is {last_time}")
        if last_time and (current_time - last_time) > timedelta(seconds=20):
            preset = camera_occupation[index]
            with camera_data_lock:
                preset_occupation[preset] = 0
                ifwork[index] = 0
                camera_occupation[index] = 0
            with last_activity_lock:
                last_activity_times[index] = None
            # logger.info(f"Camera {index + 30} released from preset {preset}")
            # logger.info(time.time())


def get_speaker_info(host, port, mark, conn):
    #termination_count = 1000
    current_count = 0
    digit_counts = {i: 0 for i in range(1, 9)}
    current_output = None
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    logger.info(f"Successfully connected to device {host}:{port} \n")

    while True:
    #while current_count < termination_count:
        get_positions_command = "< SAMPLE_TALKER_POSITIONS >"
        client_socket.send(get_positions_command.encode())
        response = client_socket.recv(1024).decode()
        # print(response)
        sample_start = response.find("< SAMPLE TALKER_POSITIONS")
        if sample_start != -1:
            sample_end = response.find(">", sample_start)
            sample_position = response[sample_start:sample_end + 1]
            if sample_position:
                numbers = [int(s) for s in sample_position.split() if s.isdigit()]
                for digit in range(1, 9):
                    if digit == numbers[1]:
                        digit_counts[digit] += 1

                max_count_digit = max(digit_counts, key=digit_counts.get)
                if current_output != max_count_digit:
                    current_output = max_count_digit
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    logger.info(f"{mark} position changed output: {current_output} - Time: {current_time}")
                    logger.info(time.time())

                    conn.send(max_count_digit)

        # time.sleep(0.6)
        current_count += 1
        for digit in range(1, 9):
            digit_counts[digit] = 0

    logger.info("Reached maximum output count, program terminated.")
    client_socket.close()

def camera(pipe1, pipe2):
    while True:
        if pipe1.poll():
            msg1 = pipe1.recv()
            
            # send_time1 = time.time()

            if msg1:
                # release_camera_start1 = time.time()
                release_camera()  # 使用摄像头后调用释放逻辑
                # release_camera_end1 = time.time()
                # logger.info(f"释放摄像头检测程序用时：{release_camera_end1 - release_camera_start1}")
                
                # start_time1 = time.time()
                use_camera(mic40[msg1]) 
                end_time1 = time.time() 
                logger.info(f"摄像头操作结束: {end_time1}")
                # logger.info(f"调用摄像头操作时间: {end_time1 - start_time1}")      
                # logger.info(f"接受时间--操作完成时间差: {end_time1 - send_time1}")

        if pipe2.poll():
            msg2 = pipe2.recv()
            
            # send_time2 = time.time()

            if msg2:
                # release_camera_start2 = time.time()
                release_camera()  # 使用摄像头后调用释放逻辑
                # release_camera_end2 = time.time()
                # logger.info(f"释放摄像头检测程序用时：{release_camera_end2 - release_camera_start2}")

                # start_time2 = time.time()
                use_camera(mic42[msg2])
                end_time2 = time.time()
                logger.info(f"摄像头操作结束: {end_time2}")
                # logger.info(f"摄像头操作时间: {end_time2 - start_time2}")
                # logger.info(f"接受时间--操作完成时间差: {end_time2 - send_time2}")

def main():
    host1, port1 = "...", ...
    host2, port2 = "...", ...   # your own IP and port
    
    thread1 = threading.Thread(target=get_speaker_info, args=(host1, port1, "绿", conn1_a))
    thread2 = threading.Thread(target=get_speaker_info, args=(host2, port2, "蓝", conn2_a))
    process_camera = Process(target=camera, args=(conn1_b, conn2_b))

    thread1.start()
    thread2.start()
    process_camera.start()

    thread1.join()
    thread2.join()
    process_camera.join()

if __name__ == "__main__":
    main()
