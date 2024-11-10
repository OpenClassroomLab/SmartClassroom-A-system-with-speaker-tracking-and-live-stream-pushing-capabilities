import numpy as np
import cv2
import torch
import sys
import os
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

class yolo_detect():
    def __init__(self, weights = ROOT / 'runs/train/exp/weights/best.pt', device = select_device('0'), dnn=False, data= ROOT / 'data/coco128.yaml'):
        self.model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=False)
        self.stridel, self.names = self.model.stride, self.model.names
        self.imgsz = check_img_size((480,640), s=self.stridel)  # check image size        
    
    def egn_warmup(self):
        self.model.warmup(imgsz=(1, 3, *self.imgsz))

    def img_detect(self, frame):
        dt = (Profile(), Profile(), Profile())
        frame = cv2.resize(frame,(self.imgsz[1],self.imgsz[0]))
        # Convert the frame from BGR to RGB
        im0s = frame
        frame = letterbox(im0s, self.imgsz, stride=self.stridel, auto=True)[0]  # padded resize
        frame = frame.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        frame = np.ascontiguousarray(frame)  # contiguous
           
        with dt[0]:
            frame = torch.from_numpy(frame).to(self.model.device)
            frame = frame.half() if self.model.fp16 else frame.float()  # uint8 to fp16/32
            frame /= 255  # 0 - 255 to 0.0 - 1.0
            if len(frame.shape) == 3:
                frame = frame[None]  # expand for batch dim 
                
        # Inference
        with dt[1]:
            pred = self.model(frame, augment=False, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=1000)
                                    
            for i, det in enumerate(pred):  # per image
                annotator = Annotator(im0s, line_width=3, example=str(self.names))
                for *xyxy, conf, cls in reversed(det):
                    annotator.box_label(xyxy, label=None, color=colors(0, True))
                if det.shape[0]!=0:
                    kuang = np.array(det[0][:4].cpu())
                else:
                    kuang = None
            frame = annotator.result()
                                   
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame,(self.imgsz[1],int(self.imgsz[0]*0.94)))
        return frame, kuang
        
#示例：frame=cv2.imread('xxx')
#      yolo=yolo_detect()
#      yolo.egn_warmup()
#      frame, kuang=yolo.img_detect(frame)
#      返回的frame为加了框的图像，kuang为ROI坐标[y0,x0,y1,x1].