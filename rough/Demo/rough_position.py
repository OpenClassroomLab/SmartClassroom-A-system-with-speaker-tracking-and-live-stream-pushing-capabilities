<<<<<<< HEAD
## 4.1 区分是哪个麦克风还没做

from multiprocessing import Process, Pipe
import time
from time import sleep

import socket

from onvif_hik import Onvif_hik
import zeep

#麦克风40
mic40=[0,0,0,0,13,11,16,14,12]


#麦克风42
mic42=[0,0,18,19,21,20,17,22,23]
    
# ___init___Pipe
conn1, conn2 = Pipe(duplex=True) #管道的两个端扣口

def micro():
    termination_count = 50
    current_count = 0

# ________start microphone________

    # 设备的IP地址和端口号
    device_ip = '...'
    device_port =   # 替换成设备的实际端口号

    # 建立TCP连接
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((device_ip, device_port))
    print(f"成功连接到设备 {device_ip}:{device_port}")

    #设置说话人位置信息反馈时间，**ms反馈一次，有效值100-9999
    set_position_command = f"< SET TALKER_POSITION_RATE {100} >"
    client_socket.send(set_position_command.encode())
    response = client_socket.recv(1024).decode()
    print(response)

    #设置完成之后再得到一次，确认真的改动成功
    get_rate_command = f"< GET TALKER_POSITION_RATE >"
    client_socket.send(get_rate_command.encode())
    response = client_socket.recv(1024).decode()
    print(response)
    
    # 自动/手动
    set_auto_coverage_off_cmd = "< SET AUTO_COVERAGE ON >"
    client_socket.send(set_auto_coverage_off_cmd.encode())
    response = client_socket.recv(1024).decode()
    print(f"自动覆盖范围的响应：{response}")
    
    
    
# ________send message________

    # 用于记录每个数字出现的次数的字典
    digit_counts = {i: 0 for i in range(1, 9)}

    # 当前输出的数字
    current_output = None

    #while current_count < termination_count:
    while True:
        get_positions_command = "< SAMPLE_TALKER_POSITIONS >"
        client_socket.send(get_positions_command.encode())
        response = client_socket.recv(1024).decode()
        print(response)
        # 定位 SAMPLE TALKER_POSITIONS 部分的开始位置
        sample_start = response.find("< SAMPLE TALKER_POSITIONS")
        

        # 如果找到了 < SAMPLE TALKER_POSITIONS，则从该位置开始查找 >
        if sample_start != -1:
            sample_end = response.find(">", sample_start)

            # 提取 SAMPLE TALKER_POSITIONS 部分
            sample_position = response[sample_start:sample_end + 1]
            if sample_position:
                # 提取数字部分
                numbers = [int(s) for s in sample_position.split() if s.isdigit()]
                print(numbers)

                # 更新数字出现次数
                for digit in range(1, 9):
                    if digit == numbers[1]:
                        digit_counts[digit] += 1
                    else:
                        digit_counts[digit] = 0

                # 获取出现次数最多的数字
                max_count_digit = max(digit_counts, key=digit_counts.get)

                # 只有当当前输出不是当前数字时，才更新输出
                if current_output != max_count_digit:
                    current_output = max_count_digit
                    #print(f"输出：{current_output}")
                #print(f"说话人位置：{sample_position}")
                conn1.send(current_output)
        time.sleep(1)  # 设置每次获取之间的等待时间，可以根据需要调整

        current_count += 1

    print("已达到输出次数上限，程序终止。")
    
# ___init___Pipe
conn1, conn2 = Pipe(duplex=True) #管道的两个端扣口   
    
def camera(x, pipe):
    o30 = Onvif_hik(ip="...", username="...", password="...")
    o31 = Onvif_hik(ip="...", username="...", password="...")
    conn1, conn2 = pipe
    while True:
        if conn2.poll():
            msg = conn2.recv()
            o30.goto_preset(mic40[msg])
            o31.goto_preset(mic40[msg])
        time.sleep(1)    
    
    
if __name__ == '__main__':
    
    sub_process = Process(target=camera, args=(100, (conn1, conn2)))
    sub_process.start()

    micro()
    
    sub_process.terminate() # 命令子进程退出
=======
## 4.1 区分是哪个麦克风还没做

from multiprocessing import Process, Pipe
import time
from time import sleep

import socket

from onvif_hik import Onvif_hik
import zeep

#麦克风40
mic40=[0,0,0,0,13,11,16,14,12]


#麦克风42
mic42=[0,0,18,19,21,20,17,22,23]
    
# ___init___Pipe
conn1, conn2 = Pipe(duplex=True) #管道的两个端扣口

def micro():
    termination_count = 50
    current_count = 0

# ________start microphone________

    # 设备的IP地址和端口号
    device_ip = '...'
    device_port =   # 替换成设备的实际端口号

    # 建立TCP连接
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((device_ip, device_port))
    print(f"成功连接到设备 {device_ip}:{device_port}")

    #设置说话人位置信息反馈时间，**ms反馈一次，有效值100-9999
    set_position_command = f"< SET TALKER_POSITION_RATE {100} >"
    client_socket.send(set_position_command.encode())
    response = client_socket.recv(1024).decode()
    print(response)

    #设置完成之后再得到一次，确认真的改动成功
    get_rate_command = f"< GET TALKER_POSITION_RATE >"
    client_socket.send(get_rate_command.encode())
    response = client_socket.recv(1024).decode()
    print(response)
    
    # 自动/手动
    set_auto_coverage_off_cmd = "< SET AUTO_COVERAGE ON >"
    client_socket.send(set_auto_coverage_off_cmd.encode())
    response = client_socket.recv(1024).decode()
    print(f"自动覆盖范围的响应：{response}")
    
    
    
# ________send message________

    # 用于记录每个数字出现的次数的字典
    digit_counts = {i: 0 for i in range(1, 9)}

    # 当前输出的数字
    current_output = None

    #while current_count < termination_count:
    while True:
        get_positions_command = "< SAMPLE_TALKER_POSITIONS >"
        client_socket.send(get_positions_command.encode())
        response = client_socket.recv(1024).decode()
        print(response)
        # 定位 SAMPLE TALKER_POSITIONS 部分的开始位置
        sample_start = response.find("< SAMPLE TALKER_POSITIONS")
        

        # 如果找到了 < SAMPLE TALKER_POSITIONS，则从该位置开始查找 >
        if sample_start != -1:
            sample_end = response.find(">", sample_start)

            # 提取 SAMPLE TALKER_POSITIONS 部分
            sample_position = response[sample_start:sample_end + 1]
            if sample_position:
                # 提取数字部分
                numbers = [int(s) for s in sample_position.split() if s.isdigit()]
                print(numbers)

                # 更新数字出现次数
                for digit in range(1, 9):
                    if digit == numbers[1]:
                        digit_counts[digit] += 1
                    else:
                        digit_counts[digit] = 0

                # 获取出现次数最多的数字
                max_count_digit = max(digit_counts, key=digit_counts.get)

                # 只有当当前输出不是当前数字时，才更新输出
                if current_output != max_count_digit:
                    current_output = max_count_digit
                    #print(f"输出：{current_output}")
                #print(f"说话人位置：{sample_position}")
                conn1.send(current_output)
        time.sleep(1)  # 设置每次获取之间的等待时间，可以根据需要调整

        current_count += 1

    print("已达到输出次数上限，程序终止。")
    
# ___init___Pipe
conn1, conn2 = Pipe(duplex=True) #管道的两个端扣口   
    
def camera(x, pipe):
    o30 = Onvif_hik(ip="...", username="...", password="...")
    o31 = Onvif_hik(ip="...", username="...", password="...")
    conn1, conn2 = pipe
    while True:
        if conn2.poll():
            msg = conn2.recv()
            o30.goto_preset(mic40[msg])
            o31.goto_preset(mic40[msg])
        time.sleep(1)    
    
    
if __name__ == '__main__':
    
    sub_process = Process(target=camera, args=(100, (conn1, conn2)))
    sub_process.start()

    micro()
    
    sub_process.terminate() # 命令子进程退出
>>>>>>> 42549d3 (准备pull到github)
    sub_process.join() # 等待子进程退出