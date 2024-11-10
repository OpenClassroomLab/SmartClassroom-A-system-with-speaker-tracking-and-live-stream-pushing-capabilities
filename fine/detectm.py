# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import time

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


@smart_inference_mode()
def run(
        weights1=ROOT / 'yolov5s.pt',  # model1 path or triton URL
        weights2=ROOT / 'yolov5s.pt',  # model2 path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download
        
    # Dataloader
    bs = 1  # batch_size
    max_fetch=100
    pixel_threshold = 5.0
    #roi=[160,400,480,520]
    roi=[180,400,490,520]
    
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    if not nosave:
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        if webcam:
            (save_dir / 'faces').mkdir(parents=True, exist_ok=True)  # make dir
            (save_dir / 'faces' / 'train').mkdir(parents=True, exist_ok=True)  # make dir
            (save_dir / 'faces' / 'test').mkdir(parents=True, exist_ok=True)  # make dir
            for j in range(max_fetch):
                (save_dir / 'faces' / 'train' / '{}'.format(j)).mkdir(parents=True, exist_ok=True)
                (save_dir / 'faces' / 'test' / '{}'.format(j)).mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = select_device(device)
    model1 = DetectMultiBackend(weights1, device=device, dnn=dnn, data=data, fp16=half)
    model2 = DetectMultiBackend(weights2, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model1.stride, model1.names, model1.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    
    #init
    vid_path, vid_writer, face_writer = [None] * bs, [None] * bs, [[None] * max_fetch] * bs
    
    mouth_writer=[[None] * max_fetch] * bs
    
    prev_cord=[[None]*2]* max_fetch
    cord=[[None]*2]* max_fetch
    
    prev_fetch=[[[[None]*imgsz[1]]*imgsz[0]]*3]* max_fetch
    fetch=[[[[None]*imgsz[1]]*imgsz[0]]*3]* max_fetch
    mouth_roi=[[[[None]*imgsz[1]]*imgsz[0]]*3]* max_fetch
    prev_gray=[[[None]*imgsz[1]]*imgsz[0]] * max_fetch
    gray=[[[None]*imgsz[1]]*imgsz[0]]* max_fetch
    prev_H=[[None]*256]* max_fetch
    H=[[None]*256]* max_fetch
    diff=[0]* max_fetch
    spk_label=[False]* max_fetch
    prev_time= [time.time()]* max_fetch
    current_time=[0]* max_fetch
    frame_num=[0]*max_fetch
    
    # Run inference
    model1.warmup(imgsz=(1 if pt or model1.triton else bs, 3, *imgsz))  # warmup
    model2.warmup(imgsz=(1 if pt or model2.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt ,dt2 = 0, [], (Profile(), Profile(), Profile()), (Profile(), Profile(), Profile())
    
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model1.device)
            im = im.half() if model1.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model1(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / 'predictions.csv'

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)


        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            ims=im0
                       
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            save_face_path=[str(save_dir / 'face0.mp4')]
            save_mouth_path=[str(save_dir / 'mouth0.mp4')]
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                
                #fetch objects                
                fetch_unsorted=[cv2.resize(ims[int(det[l,1]):int(det[l,3]),int(det[l,0]):int(det[l,2]),:],imgsz) for l in range(len(det))]
                fetch=fetch_unsorted.copy()
                cord_unsorted=np.array([[int(det[l,1])+int(det[l,3]),int(det[l,1])+int(det[l,3])] for l in range(len(det))])
                cord=cord_unsorted.copy()
                dist=np.array([[0.0]*len(prev_cord)]*len(cord_unsorted))
                simi=np.zeros(len(cord_unsorted),dtype=int)                
                
                #trace by swap
                if prev_cord[0][0]!=None:
                    for u in range(len(cord_unsorted)):
                        dist[u]=np.array([np.linalg.norm(prev_cord[l]-cord_unsorted[u]) for l in range(len(prev_cord))])
                        simi[u]=np.argmin(dist[u])
                    if len(cord_unsorted)==len(prev_cord):
                        for u in range(len(cord_unsorted)):
                            fetch[simi[u]]=fetch_unsorted[u]
                            cord[simi[u]]=cord_unsorted[u]
                    if len(cord_unsorted)>len(prev_cord):
                        indices = np.sort(np.argsort(simi)[:len(prev_cord)])
                        leak_indices=np.setdiff1d(np.array(range(len(cord))),indices)
                        simi_cut = simi[indices]
                        for u in range(len(prev_cord)):
                            fetch[simi_cut[u]]=fetch_unsorted[indices[u]]
                            cord[simi_cut[u]]=cord_unsorted[indices[u]]
                        for p in range(u+1,len(cord_unsorted)):
                            fetch[p]=fetch_unsorted[leak_indices[p-u-1]]
                            cord[p]=cord_unsorted[leak_indices[p-u-1]]
                    if len(cord_unsorted)<len(prev_cord):
                        fetch=prev_fetch.copy()
                        cord=prev_cord.copy()
                        for u in range(len(cord_unsorted)):
                            fetch[simi[u]]=fetch_unsorted[u]
                            cord[simi[u]]=cord_unsorted[u]                
                prev_fetch=fetch.copy()
                prev_cord=cord.copy()
                
                #mouth detection
                if webcam:
                    for k in range(len(det)):
                        
                        fetchc=[fetch[k]]
                        fetchcpy = np.stack([letterbox(x, imgsz, stride=stride, auto=pt)[0] for x in fetchc])  # resize
                        fetchcpy = fetchcpy[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
                        fetchcpy = np.ascontiguousarray(fetchcpy)  # contiguous
                        with dt2[0]:
                            fetchcpy = torch.from_numpy(fetchcpy).to(model2.device)
                            fetchcpy = fetchcpy.half() if model2.fp16 else fetchcpy.float()  # uint8 to fp16/32
                            fetchcpy /= 255  # 0 - 255 to 0.0 - 1.0
                            if len(fetchcpy.shape) == 3:
                                fetchcpy = fetchcpy[None]  # expand for batch dim
                        # Inference
                        with dt2[1]:
                            pred2 = model2(fetchcpy, augment=augment, visualize=visualize)
                        # NMS
                        with dt2[2]:
                            pred2 = non_max_suppression(pred2, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                        if len(pred2[0]):
                            det2=pred2[0]
                            det2[:, :4] = scale_boxes(fetchcpy.shape[2:], det2[:, :4], fetch[k].shape).round()
                            mouth_roi[k]=cv2.resize(fetch[k][int(det2[0,1]):int(det2[0,3]),int(det2[0,0]):int(det2[0,2]),:],imgsz)
                        if mouth_roi[k][0][0][0]==None:
                            mouth_roi[k]=cv2.resize(fetch[k][roi[1]:roi[3],roi[0]:roi[2],:],imgsz)

                        gray[k] = cv2.equalizeHist(cv2.cvtColor(mouth_roi[k], cv2.COLOR_BGR2GRAY))
                        #gray[k] = cv2.adaptiveThreshold(gray[k],1,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
                        #gray[k] = np.array(gray[k])
                        H[k]=cv2.calcHist(gray[k], [0], None, [256], [0, 256])
                        
                        if prev_gray[k][0][0]!=None:
                            
                            diff[k]+= cv2.compareHist(H[k],prev_H[k], 0)                       
                            current_time[k] = time.time() 
                            if current_time[k] - prev_time[k] >= 2:
                                prev_time[k] = current_time[k]
                                if diff[k] < pixel_threshold:
                                    spk_label[k]=True
                                else:
                                    spk_label[k]=False
                                diff[k] = 0                           
                    prev_gray = gray.copy()
                    prev_H=H.copy()
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                cont=0
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else ('speaking' if spk_label[cont] else (names[c] if hide_conf else f'{names[c]} {conf:.2f}'))
                        annotator.box_label(xyxy, label, color=colors(5, True) if spk_label[cont] else colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    cont+=1
                    
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    #cv2.imwrite(save_path, im0)
                    for i in range(len(fetch)):
                        cv2.imwrite(str(save_dir / '{}_fetch{}.jpg'.format(p.name,i)), fetch[i])
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        for j in range(max_fetch):
                            if isinstance(face_writer[i][j], cv2.VideoWriter):
                                face_writer[i][j].release()  # release previous video writer
                            if isinstance(mouth_writer[i][j], cv2.VideoWriter):
                                mouth_writer[i][j].release()  # release previous video writer                            
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        save_face_path=[str(Path(str(save_dir / 'face{}'.format(j))).with_suffix('.mp4')) for j in range(max_fetch)]
                        save_mouth_path=[str(Path(str(save_dir / 'mouth{}'.format(j))).with_suffix('.mp4')) for j in range(max_fetch)]                        
                        for j in range(max_fetch):    
                            face_writer[i][j] = cv2.VideoWriter(save_face_path[j], cv2.VideoWriter_fourcc(*'mp4v'), fps, imgsz)
                            mouth_writer[i][j] = cv2.VideoWriter(save_mouth_path[j], cv2.VideoWriter_fourcc(*'mp4v'), fps, imgsz)
                    vid_writer[i].write(im0)
                    for j in range(len(fetch)):
                        if(face_writer[i][j]!=None) and fetch[j][0][0][0]!=None:
                            face_writer[i][j].write(fetch[j])
                        if(mouth_writer[i][j]!=None) and fetch[j][0][0][0]!=None:
                            mouth_writer[i][j].write(mouth_roi[j])
                            
                    #save frames
                    save_img_train=[str(Path(str(save_dir / 'faces' / 'train' / '{}'.format(j) / '{}'.format(frame_num[j]))).with_suffix('.jpg')) for j in range(max_fetch)]
                    save_img_test=[str(Path(str(save_dir / 'faces' / 'test' / '{}'.format(j) / '{}'.format(frame_num[j]))).with_suffix('.jpg')) for j in range(max_fetch)]
                    for j in range(len(fetch)):
                        if frame_num[j]<5:
                            cv2.imwrite(save_img_test[j], fetch[j])
                        else:
                            cv2.imwrite(save_img_train[j], fetch[j])                        
                        frame_num[j]+=1
                        

        # Print time (inference-only)
        LOGGER.info(f"{s} diff= {diff[0]}  {'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        
    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    return save_dir,max_fetch,(save_img and webcam)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights1', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--weights2', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def cut(save_dir,max_fetch):
    for i in range(max_fetch):
        face_path=str(Path(str(save_dir / 'face{}'.format(i))).with_suffix('.mp4'))
        mouth_path=str(Path(str(save_dir / 'mouth{}'.format(i))).with_suffix('.mp4'))
        save_img_train=str(Path(str(save_dir / 'faces' / 'train' / '{}'.format(i))))
        save_img_test=str(Path(str(save_dir / 'faces' / 'test' / '{}'.format(i))))
        if os.path.getsize(face_path)<512:
            os.remove(face_path)
        if os.path.getsize(mouth_path)<512:
            os.remove(mouth_path)
        if len(os.listdir(save_img_train))==0:
            os.removedirs(save_img_train)
        if len(os.listdir(save_img_test))==0:
            os.removedirs(save_img_test)

#def classify(save_dir,)
def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    save_dir,max_fetch,save_cam=run(**vars(opt))
    if save_cam:     
        cut(save_dir,max_fetch)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
