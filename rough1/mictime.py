
from multiprocessing import Process, Pipe
import time
from time import sleep
import socket

def micro():
    # 设备的IP地址和端口号
    device_ip = '...'
    device_port = ...  # 替换成设备的实际端口号



    # 建立TCP连接
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((device_ip, device_port))
    print(f"成功连接到设备 {device_ip}:{device_port}")


if __name__ == '__main__':
    
    start_time = time.time()
    micro()
    end_time = time.time()
    print(end_time - start_time)