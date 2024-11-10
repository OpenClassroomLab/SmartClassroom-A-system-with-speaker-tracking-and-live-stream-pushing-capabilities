import tkinter as tk
from PIL import Image, ImageTk

import numpy as np
import torch
import time
import threading
import os
import platform
import sys
from pathlib import Path

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

class VideoStreamApp:

    def __init__(self, rtsp_url):
        self.root = tk.Tk()
        self.root.title("RTSP Video Stream")

        self.root.resizable(0, 0)
        self.root.overrideredirect(True)
        
        sw = self.root.winfo_screenwidth()
        # 得到屏幕宽度
        sh = self.root.winfo_screenheight()
        # 得到屏幕高度

        # 窗口宽高
        (wh,ww)=imgsz
        x = (sw - ww) / 2
        y = (sh - wh) / 2
        self.root.geometry("%dx%d+%d+%d" % (ww, wh, x, y))

        # 创建退出按键
        self.button = tk.Button(self.root, text='退出', command=self.root.quit)
        self.button.pack()
        
        self.canvas = tk.Canvas(self.root, width=ww, height=wh)
        self.canvas.pack()

        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(rtsp_url)
        self.thread = threading.Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame,(imgsz[1],imgsz[0]))
                # Convert the frame from BGR to RGB
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
                                                   
                    frame = annotator.result()
                        
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert the frame to ImageTk format
                img = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(image=img)

                # Update the canvas with the new frame
                self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                self.canvas.img = img_tk  # Keep a reference to avoid garbage collection

    def on_close(self):
        self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    rtsp_url = "..."  # the rtsp url of your device]
    app = VideoStreamApp(rtsp_url)
    app.run()