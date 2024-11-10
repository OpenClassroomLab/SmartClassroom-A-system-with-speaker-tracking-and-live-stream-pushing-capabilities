import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import time

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox

weights='D:/yolov5/runs/train/exp/weights/best.pt'
device = select_device('0')
dnn=False
data='D:/yolov5/data/coco128.yaml'
    
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((1080,1920), s=stride)  # check image size
model.warmup(imgsz=(1, 3, *imgsz))
seen, windows, dt ,dt2 = 0, [], (Profile(), Profile(), Profile()), (Profile(), Profile(), Profile())

dataset = LoadImages('D:/test_stamp[600].jpg', img_size=imgsz, stride=stride, auto=pt, vid_stride=1)

for path, frame, im0s, vid_cap, s in dataset:
    with dt[0]:
        frame = torch.from_numpy(frame).to(model.device)
        print('size=',frame.shape)
        frame = frame.half() if model.fp16 else frame.float()  # uint8 to fp16/32
        frame /= 255  # 0 - 255 to 0.0 - 1.0
        if len(frame.shape) == 3:
            frame = frame[None]  # expand for batch dim
        print('size=',frame.shape)               
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
                                       
        im0 = annotator.result()
        cv2.imwrite('D:/yolosave.jpg', im0) 
        kuang = np.array(det[0][:4].cpu())
        print('kuang=',kuang)