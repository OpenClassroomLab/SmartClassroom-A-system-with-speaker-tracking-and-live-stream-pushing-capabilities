# coding=utf-8

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import platform
import tkinter
from tkinter import *
from HCNetSDK import *
from PlayCtrl import *
import time
import cv2
from PIL import Image, ImageTk


import numpy as np
import torch
import sys
from pathlib import Path
from filterpy.kalman import KalmanFilter
import math

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox

weights = ROOT / 'runs/train/exp/weights/best.pt'
device = select_device('0')
dnn=False
data= ROOT / 'data/coco128.yaml'
    
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=False)
stridel, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((480,640), s=stridel)  # check image size
model.warmup(imgsz=(1, 3, *imgsz))
seen, windows, dt ,dt2 = 0, [], (Profile(), Profile(), Profile()), (Profile(), Profile(), Profile())

time0=time.time()

# 登录的设备信息
DEV_IP = create_string_buffer(b'10...')  # put camera IP address
DEV_PORT = 8000
DEV_USER_NAME = create_string_buffer(b'...') # admin
DEV_PASSWORD = create_string_buffer(b'...') # password

WINDOWS_FLAG = True
win = None  # 预览窗口
funcRealDataCallBack_V30 = None  # 实时预览回调函数，需要定义为全局的

PlayCtrl_Port = c_long(-1)  # 播放句柄
Playctrldll = None  # 播放库
FuncDecCB = None   # 播放库解码回调函数，需要定义为全局的

i1 = 0
crop            = False
count           = False
history_centers = []  # 存储历史中心点坐标
max_history = 10      # 最大历史记录数，可以根据需要调整


mode=1   #mode=1代表跟踪，mode=0代表预测

def yv12_to_rgb(yv12_data, width, height):
    # YV12格式的图像分为Y、V、U三个平面
    # Y平面存储亮度信息，V和U平面存储色度信息
    # 以下计算得到各个平面在内存中的起始位置
    size_y = width * height
    size_uv = size_y // 4
    y_data = yv12_data[:size_y]
    v_data = yv12_data[size_y:size_y + size_uv]
    u_data = yv12_data[size_y + size_uv:]

    # 将Y、V、U数据转换为numpy数组
    y_plane = np.frombuffer(y_data, dtype=np.uint8).reshape((height, width))
    v_plane = np.frombuffer(v_data, dtype=np.uint8).reshape((height // 2, width // 2))
    u_plane = np.frombuffer(u_data, dtype=np.uint8).reshape((height // 2, width // 2))
    
    Y = y_plane.astype(np.float32)
    U = u_plane.repeat(2, axis=0).repeat(2, axis=1).astype(np.float32)
    V = v_plane.repeat(2, axis=0).repeat(2, axis=1).astype(np.float32)

    R = Y + 1.403 * (V - 128)
    G = Y - 0.344 * (U - 128) - 0.714 * (V - 128)
    B = Y + 1.770 * (U - 128)

    R = np.clip(R, 0, 255).astype(np.uint8)
    G = np.clip(G, 0, 255).astype(np.uint8)
    B = np.clip(B, 0, 255).astype(np.uint8)

    return cv2.merge([B, G, R])
    
def calculate_dynamic_sleep(diff_x, diff_y, max_diff, min_sleep, max_sleep):
    # 计算总的距离
    distance = math.sqrt(diff_x ** 2 + diff_y ** 2)

    # 归一化距离并映射到sleep时间
    sleep_time = min(distance/ 400 * 0.1, max_sleep) 
    print(sleep_time)
    return sleep_time

def initialize_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0., 0., 0., 0.])  # 初始状态 (位置和速度)
    kf.F = np.array([[1, 0, 1, 0],    # 状态转移矩阵
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],    # 测量函数
                     [0, 1, 0, 0]])
    kf.P *= 1000.                    # 协方差矩阵
    kf.R = np.array([[1, 0],         # 测量噪声
                     [0, 1]])
    kf.Q = np.eye(kf.dim_x) * 0.1    # 过程噪声

    return kf

def predict_next_position_kalman(history_centers, kf):
    if not history_centers:
        return None

    last_center = history_centers[-1]
    kf.predict()
    kf.update(last_center)

    predicted = kf.x[:2]  # 预测的位置
    return int(predicted[0]), int(predicted[1])

# 初始化卡尔曼滤波器
kf = initialize_kalman_filter()


def draw_trajectory_on_image(image, history_centers, predicted_position):
    # 历史点颜色和尺寸
    history_color = (255, 0, 0)  # 蓝色
    history_size = 3

    # 预测点颜色和尺寸
    predicted_color = (0, 0, 255)  # 红色
    predicted_size = 5

    # 绘制历史轨迹点
    for center in history_centers:
        center_int = (int(center[0]), int(center[1]))
        cv2.circle(image, center_int, history_size, history_color, -1)

    # 绘制轨迹线
    for i in range(1, len(history_centers)):
        cv2.line(image, history_centers[i - 1], history_centers[i], history_color, 2)

    # 绘制预测点
    if predicted_position is not None:
        predicted_position=(int(predicted_position[0]), int(predicted_position[1]))
        cv2.circle(image, predicted_position, predicted_size, predicted_color, -1)
        if history_centers:
            # 从最后一个历史点连线到预测点
            cv2.line(image, history_centers[-1], predicted_position, predicted_color, 2)

    return image

prev_frame_time = 0
new_frame_time = 0

# 创建窗口
win = tkinter.Tk()
#固定窗口大小
win.resizable(0, 0)
win.overrideredirect(True)

sw = win.winfo_screenwidth()
# 得到屏幕宽度
sh = win.winfo_screenheight()
# 得到屏幕高度

# 窗口宽高
(wh,ww)=imgsz
x = (sw - ww) / 2
y = (sh - wh) / 2
win.geometry("%dx%d+%d+%d" % (ww, wh, x, y))

# 创建退出按键
b = Button(win, text='Quit', command=win.quit)
b.pack()
# 创建一个Canvas，设置其背景色为白色
cv = tkinter.Canvas(win, bg='white', width=ww, height=wh)
cv.pack()
    
# 获取当前系统环境
def GetPlatform():
    sysstr = platform.system()
    print('' + sysstr)
    if sysstr != "Windows":
        global WINDOWS_FLAG
        WINDOWS_FLAG = False

# 设置SDK初始化依赖库路径
def SetSDKInitCfg():
    # 设置HCNetSDKCom组件库和SSL库加载路径
    # print(os.getcwd())
    if WINDOWS_FLAG:
        strPath = os.getcwd().encode('gbk')
        sdk_ComPath = NET_DVR_LOCAL_SDK_PATH()
        sdk_ComPath.sPath = strPath
        Objdll.NET_DVR_SetSDKInitCfg(2, byref(sdk_ComPath))
        Objdll.NET_DVR_SetSDKInitCfg(3, create_string_buffer(strPath + b'\libcrypto-1_1-x64.dll'))
        Objdll.NET_DVR_SetSDKInitCfg(4, create_string_buffer(strPath + b'\libssl-1_1-x64.dll'))
    else:
        strPath = os.getcwd().encode('utf-8')
        sdk_ComPath = NET_DVR_LOCAL_SDK_PATH()
        sdk_ComPath.sPath = strPath
        Objdll.NET_DVR_SetSDKInitCfg(2, byref(sdk_ComPath))
        Objdll.NET_DVR_SetSDKInitCfg(3, create_string_buffer(strPath + b'/libcrypto.so.1.1'))
        Objdll.NET_DVR_SetSDKInitCfg(4, create_string_buffer(strPath + b'/libssl.so.1.1'))

def LoginDev(Objdll):
    # 登录注册设备
    device_info = NET_DVR_DEVICEINFO_V30()
    lUserId = Objdll.NET_DVR_Login_V30(DEV_IP, DEV_PORT, DEV_USER_NAME, DEV_PASSWORD, byref(device_info))
    return (lUserId, device_info)

def DecCBFun(nPort, pBuf, nSize, pFrameInfo, nUser, nReserved2):
    # 解码回调函数
    if pFrameInfo.contents.nType == 3:
        # 解码返回视频YUV数据，将YUV数据转成jpg图片保存到本地
        # 如果有耗时处理，需要将解码数据拷贝到回调函数外面的其他线程里面处理，避免阻塞回调导致解码丢帧
        sFileName = ('../../pic/test_stamp[%d].jpg'% pFrameInfo.contents.nStamp)
        nWidth = pFrameInfo.contents.nWidth
        nHeight = pFrameInfo.contents.nHeight
        nType = pFrameInfo.contents.nType
        dwFrameNum = pFrameInfo.contents.dwFrameNum
        nStamp = pFrameInfo.contents.nStamp
        print(nWidth, nHeight, nType, dwFrameNum, nStamp)

        global time0
        global kuang
        time1=time.time()

        if time1-time0 > 0.1:
            buffer = ctypes.create_string_buffer(nSize)
            ctypes.memmove(buffer, pBuf, nSize)
            yv12_data=np.frombuffer(buffer.raw, dtype=np.uint8).reshape(int(1.5*nHeight), nWidth)
            print('shape=',yv12_data.shape)
            frame = cv2.cvtColor(yv12_data, cv2.COLOR_YUV420p2BGR)
            '''
            cv2.imshow("Demo2", frame)
            key = cv2.waitKey(0)
            Playctrldll.PlayM4_Pause(PlayCtrl_Port, 1)
            '''
            time0 = time1
            if frame is not None:
                frame = cv2.resize(frame,(imgsz[1],imgsz[0]))
                im0s = frame
                frame = letterbox(im0s, imgsz, stride=stridel, auto=True)[0]  # padded resize
                frame = frame.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                frame = np.ascontiguousarray(frame)  # contiguous
                   
                with dt[0]:
                    frame = torch.from_numpy(frame).to(model.device)
                    frame = frame.half() if model.fp16 else frame.float()  # uint8 to fp16/32
                    frame /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(frame.shape) == 3:
                        frame = frame[None]  # expand for batch dim 
                        
                # Inference
                with dt[1]:
                    pred = model(frame, augment=False, visualize=False)

                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=1000)
                                            
                    for i, det in enumerate(pred):  # per image
                        annotator = Annotator(im0s, line_width=3, example=str(names))
                        for *xyxy, conf, cls in reversed(det):
                            annotator.box_label(xyxy, label=None, color=colors(0, True))
                        if det.shape[0]!=0:
                            kuang = np.array(det[0][:4].cpu())
                        else:
                            kuang = None
                    frame = annotator.result()
                                           
                
                if kuang is not None:
                   # 计算框的中心点
                    box_center_x = (kuang[0] + kuang[2]) / 2
                    box_center_y = (kuang[1] + kuang[3]) / 2
                    
                    # 将新的中心点坐标添加到历史记录中
                    history_centers.append((int(box_center_x),int(box_center_y) ))

                    # 保持历史记录长度
                    if len(history_centers) > max_history:
                        history_centers.pop(0)


                    if mode:
                        # 计算图像的中心点
                        img_center_x = frame.shape[1] / 2
                        img_center_y = frame.shape[0] / 2

                        # 设置移动的阈值
                        threshold = 20  # 可以根据需要调整这个值

                        # 计算框中心与图像中心的差异
                        diff_x = box_center_x - img_center_x
                        diff_y = box_center_y - img_center_y

                        # 根据位置差异选择控制命令
                        command = None
                        if abs(diff_x) > threshold or abs(diff_y) > threshold:
                            dynamic_sleep = calculate_dynamic_sleep(diff_x, diff_y, 200, 0.1, 0.5)  # 参数可根据需要调整
                            if diff_x < -threshold and diff_y < -threshold:
                                command = UP_LEFT
                            elif diff_x > threshold and diff_y < -threshold:
                                command = UP_RIGHT
                            elif diff_x < -threshold and diff_y > threshold:
                                command = DOWN_LEFT
                            elif diff_x > threshold and diff_y > threshold:
                                command = DOWN_RIGHT
                            elif diff_x < -threshold:
                                command = PAN_LEFT
                            elif diff_x > threshold:
                                command = PAN_RIGHT
                            elif diff_y < -threshold:
                                command = TILT_UP
                            elif diff_y > threshold:
                                command = TILT_DOWN
                        # 控制摄像头移动
                        if command is not None:
                            Objdll.NET_DVR_PTZControl(lRealPlayHandle, command, 0)
                            time.sleep(dynamic_sleep)
                            Objdll.NET_DVR_PTZControl(lRealPlayHandle, command, 1)
                            # //TILT_UP            21    /* 云台以SS的速度上仰 */
                            # //TILT_DOWN        22    /* 云台以SS的速度下俯 */
                            # //PAN_LEFT        23    /* 云台以SS的速度左转 */
                            # //PAN_RIGHT        24    /* 云台以SS的速度右转 */
                            # //UP_LEFT            25    /* 云台以SS的速度上仰和左转 */
                            # //UP_RIGHT        26    /* 云台以SS的速度上仰和右转 */
                            # //DOWN_LEFT        27    /* 云台以SS的速度下俯和左转 */
                            # //DOWN_RIGHT        28    /* 云台以SS的速度下俯和右转 */
                            # //PAN_AUTO        29    /* 云台以SS的速度左右自动扫描 */
                    else:
                        # 调用函数进行预测
                        predicted_position = predict_next_position_kalman(history_centers, kf)
                        frame = draw_trajectory_on_image(frame, history_centers, predicted_position)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame,(imgsz[1],int(imgsz[0]*0.94)))
                    img = Image.fromarray(frame)
                    img_tk = ImageTk.PhotoImage(image=img)

                    # Update the canvas with the new frame
                    cv.create_image(0, 0, anchor=tkinter.NW, image=img_tk)
                    cv.img = img_tk  # Keep a reference to avoid garbage collection
def RealDataCallBack_V30(lPlayHandle, dwDataType, pBuffer, dwBufSize, pUser):
    # 码流回调函数
    if dwDataType == NET_DVR_SYSHEAD:
        # 设置流播放模式
        Playctrldll.PlayM4_SetStreamOpenMode(PlayCtrl_Port, 0)
        # 打开码流，送入40字节系统头数据
        if Playctrldll.PlayM4_OpenStream(PlayCtrl_Port, pBuffer, dwBufSize, 1024*1024):
            # 设置解码回调，可以返回解码后YUV视频数据
            global FuncDecCB
            FuncDecCB = DECCBFUNWIN(DecCBFun)
            #Playctrldll.PlayM4_SetDecodeEngineEx(PlayCtrl_Port, 1)
            Playctrldll.PlayM4_SetDecCallBackExMend(PlayCtrl_Port, FuncDecCB, None, 0, None)
            
            # 开始解码播放            
            if Playctrldll.PlayM4_Play(PlayCtrl_Port, cv.winfo_id()):
                print(u'播放库播放成功')                    
            else:
                print(u'播放库播放失败')                    
            
        else:
            print(u'播放库打开流失败')
    elif dwDataType == NET_DVR_STREAMDATA:
        Playctrldll.PlayM4_InputData(PlayCtrl_Port, pBuffer, dwBufSize)
    else:
        print (u'其他数据,长度:', dwBufSize)

def OpenPreview(Objdll, lUserId, callbackFun):
    '''
    打开预览
    '''
    preview_info = NET_DVR_PREVIEWINFO()
    preview_info.hPlayWnd = 0
    preview_info.lChannel = 1  # 通道号
    preview_info.dwStreamType = 0  # 主码流
    preview_info.dwLinkMode = 0  # TCP
    preview_info.bBlocked = 1  # 阻塞取流

    # 开始预览并且设置回调函数回调获取实时流数据
    lRealPlayHandle = Objdll.NET_DVR_RealPlay_V40(lUserId, byref(preview_info), callbackFun, None)
    return lRealPlayHandle

def InputData(fileMp4, Playctrldll):
    while True:
        pFileData = fileMp4.read(4096)
        if pFileData is None:
            break

        if not Playctrldll.PlayM4_InputData(PlayCtrl_Port, pFileData, len(pFileData)):
            break

if __name__ == '__main__':
    
    # 获取系统平台
    GetPlatform()

    # 加载库,先加载依赖库
    if WINDOWS_FLAG:
        os.chdir(r'./lib/win')
        Objdll = ctypes.CDLL(r'./HCNetSDK.dll')  # 加载网络库
        Playctrldll = ctypes.CDLL(r'./PlayCtrl.dll')  # 加载播放库
    else:
        os.chdir(r'./lib/linux')
        Objdll = cdll.LoadLibrary(r'./libhcnetsdk.so')
        Playctrldll = cdll.LoadLibrary(r'./libPlayCtrl.so')

    SetSDKInitCfg()  # 设置组件库和SSL库加载路径

    # 初始化DLL
    Objdll.NET_DVR_Init()
    # 启用SDK写日志
    Objdll.NET_DVR_SetLogToFile(3, bytes('./SdkLog_Python/', encoding="utf-8"), False)
   
    # 获取一个播放句柄
    if not Playctrldll.PlayM4_GetPort(byref(PlayCtrl_Port)):
        print(u'获取播放库句柄失败')

    # 登录设备
    (lUserId, device_info) = LoginDev(Objdll)
    if lUserId < 0:
        err = Objdll.NET_DVR_GetLastError()
        print('Login device fail, error code is: %d' % Objdll.NET_DVR_GetLastError())
        # 释放资源
        Objdll.NET_DVR_Cleanup()
        exit()

    # 定义码流回调函数
    funcRealDataCallBack_V30 = REALDATACALLBACK(RealDataCallBack_V30)
    # 开启预览
    lRealPlayHandle = OpenPreview(Objdll, lUserId, funcRealDataCallBack_V30)
    if lRealPlayHandle < 0:
        print ('Open preview fail, error code is: %d' % Objdll.NET_DVR_GetLastError())
        # 登出设备
        Objdll.NET_DVR_Logout(lUserId)
        # 释放资源
        Objdll.NET_DVR_Cleanup()
        exit()

    #show Windows
    win.mainloop()



    # 关闭预览
    Objdll.NET_DVR_StopRealPlay(lRealPlayHandle)

    # 停止解码，释放播放库资源
    if PlayCtrl_Port.value > -1:
        Playctrldll.PlayM4_Stop(PlayCtrl_Port)
        Playctrldll.PlayM4_CloseStream(PlayCtrl_Port)
        Playctrldll.PlayM4_FreePort(PlayCtrl_Port)
        PlayCtrl_Port = c_long(-1)

    # 登出设备
    Objdll.NET_DVR_Logout(lUserId)

    # 释放资源
    Objdll.NET_DVR_Cleanup()

