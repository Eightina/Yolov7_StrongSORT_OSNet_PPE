# python track.py --show-vid --save-vid --save-txt --source inference/short_test.mp4 
# python track.py --show-vid --source inference/short_test3.mp4 --classes 0 1 --save-vid
import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from numpy import random


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from my_utils.scaler import kpts_scale_coords
from my_utils.person import Person
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages, LoadStreams
from yolov7.utils.general import (check_img_size, non_max_suppression, non_max_suppression_kpt, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, check_file)
from yolov7.utils.torch_utils import select_device, time_synchronized
from yolov7.utils.plots import (plot_one_box, output_to_keypoint, plot_skeleton_kpts)
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT





VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes


@torch.no_grad()
def run(
        source='0',
        yolo_pose_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        yolo_ppe_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        # max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=True,  # must use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_pose_weights, list):  # single yolo model
        exp_name = yolo_pose_weights.stem
    # if not isinstance(yolo_ppe_weights, list):  # single yolo model
    #     exp_name = yolo_ppe_weights.stem
    # elif type(yolo_pose_weights) is list and len(yolo_pose_weights) == 1:  # single models after --yolo_pose_weights
    #     exp_name = Path(yolo_pose_weights[0]).stem
    #     yolo_pose_weights = Path(yolo_pose_weights[0])
    # else:  # multiple models after --yolo_pose_weights
    #     exp_name = 'ensemble'
    
    exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    save_dir = Path(save_dir)
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    
    ppe_model = attempt_load(Path(yolo_ppe_weights), map_location=device)  # load FP32 model
    pose_weights = torch.load(Path(yolo_pose_weights))
    pose_model = pose_weights['model']
    pose_model = pose_model.half().to(device)
    ppe_model = ppe_model.half().to(device)
    _ = pose_model.eval()
    
    names, = pose_model.names,
    stride = pose_model.stride.max()  # pose_model stride
    imgsz = check_img_size(imgsz[0], s=stride.cpu().numpy())  # check image size

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride.cpu().numpy())
        nr_sources = 1
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                half,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
            )
        )
        strongsort_list[i].model.warmup()
    pose_outputs = [None] * nr_sources
    
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(4)]

    # Run tracking
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap) in enumerate(dataset):
        s = ''
        t1 = time_synchronized()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_synchronized()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        ppe_pred = ppe_model(im)
        pose_pred, _ = pose_model(im)
        t3 = time_synchronized()
        dt[1] += t3 - t2

        # Apply NMS
        ppe_pred = non_max_suppression(ppe_pred[0], conf_thres, iou_thres, classes, agnostic_nms)
        pose_pred = non_max_suppression_kpt(pose_pred, 0.25, 0.65, nc=pose_model.yaml['nc'], nkpt=pose_model.yaml['nkpt'], kpt_label=True)
        dt[2] += time_synchronized() - t3
        
        # pred shape: (n_person, 58) 58 = 7 + 17 * 3
        # plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
        # like [batch_id, class_id, x, y, w, h, conf, xi, yi, confi, ......] 
            
        # Process pose detections
        # for i, pose_det in enumerate(pose_pred):  # detections per image
        # !disable ensemble! so the list pred contains only one pose_det
        i = 0
        pose_det = pose_pred[0]
        ppe_det = ppe_pred[0]
        seen += 1
        if webcam:  # nr_sources >= 1
            p, im0, _ = path[i], im0s[i].copy(), dataset.count
            p = Path(p)  # to Path
            s += f'{i}: '
            txt_file_name = p.name
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
        else:
            p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            # video file
            if source.endswith(VID_FORMATS):
                txt_file_name = p.stem
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            # folder with imgs
            else:
                txt_file_name = p.parent.name  # get folder name containing current img
                save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
                
        curr_frames[i] = im0

        txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
        s += '%gx%g ' % im.shape[2:]  # print string
        imc = im0.copy() if save_crop else im0  # for save_crop

        if cfg.STRONGSORT.ECC:  # camera motion compensation
            strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

        if pose_det is not None and len(pose_det):
            # Rescale boxes from img_size to im0 size
            pose_det[:, :4] = scale_coords(im.shape[2:], pose_det[:, :4], im0.shape).round()
            
            # do coords scaling for skeleton coords
            with torch.no_grad():
                skeleton_pred = torch.tensor(output_to_keypoint(pose_pred))
                skeleton_pred[:, 7:] = kpts_scale_coords(im.shape[2:], skeleton_pred[:, 7:], im0.shape).round()
            
            # if there is no skeleton det, then ppe det is pointless
            if ppe_det is not None and len(ppe_det):
                # Rescale boxes from img_size to im0 size
                ppe_det[:, :4] = scale_coords(im.shape[2:], ppe_det[:, :4], im0.shape).round()

            # Print results
            for c in pose_det[:, -1].unique():
                n = (pose_det[:, -1] == c).sum()  # detections per class
                s = s + f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            for c in ppe_det[:, -1].unique():
                n = (ppe_det[:, -1] == c).sum()  # detections per class
                # s_ppe = s + f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            
            xyxys_pose = pose_det[:, 0:4]
            xywhs_pose = xyxy2xywh(xyxys_pose)
            confs_pose = pose_det[:, 4]
            clss_pose = pose_det[:, 5]
            
            xyxys_ppe = ppe_det[:, 0:4]
            xywhs_ppe = xyxy2xywh(xyxys_ppe)
            confs_ppe = ppe_det[:, 4]
            clss_ppe= ppe_det[:, 5]

            # pass detections to strongsort and get bbox coords(already scaled before)
            t4 = time_synchronized()
            pose_outputs[i] = strongsort_list[i].update(xywhs_pose.cpu(), confs_pose.cpu(), clss_pose.cpu(), im0)
            t5 = time_synchronized()
            dt[3] += t5 - t4

            person_state = {}
            # draw boxes for visualization
            if len(pose_outputs[i]) > 0:
                for j, (pose_output, conf) in enumerate(zip(pose_outputs[i], confs_pose)): #output is xyxy

                    bbox = pose_output[0:4] # xyxy
                    id = int(pose_output[4])
                    cls = int(pose_output[5])
                    skeleton_coords = skeleton_pred[j, 7:].T
                    person_state[id] = Person(id = int(id), bbox = bbox, skeleton_coords = skeleton_coords.T)
                    cur_person = person_state[id]
                    
                    #detect if with ppe
                    for _, (xyxy, clss) in enumerate(zip(xyxys_ppe, clss_ppe)):
                        cur_person.ppe_paring(ppe_clss = int(clss), ppe_bbox = xyxy)
                            
                    if save_txt:
                        # to MOT format
                        bbox_left = pose_output[0]
                        bbox_top = pose_output[1]
                        bbox_w = pose_output[2] - pose_output[0]
                        bbox_h = pose_output[3] - pose_output[1]
                        # Write MOT compliant results to file
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                            bbox_top, bbox_w, bbox_h, -1, -1, -1, i))
                            # f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                            #                                 bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                    if save_vid or save_crop or show_vid:  # Add bbox to image
                        c = int(cls)  # integer class
                        id = int(id)  # integer id
                        h = cur_person.with_helmet
                        v = cur_person.with_vest
                        label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                            (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f} h:{str(h)} v:{str(v)}'))
                        plot_one_box(bbox, im0, label=label, color=colors[3], line_thickness=2)
                        
                        if cur_person.head_bbox != None:
                            plot_one_box(cur_person.head_bbox, im0, label='head', color=colors[2], line_thickness=2)
                        if cur_person.body_bbox != None:
                            plot_one_box(cur_person.body_bbox, im0, label='body', color=colors[2], line_thickness=2)
                        
                        
                        if h:
                            plot_one_box(cur_person.helmet_bbox, im0, label=f'h {cur_person.helmet_iou:.2f}', color=colors[0], line_thickness=2)
                        if v:    
                            plot_one_box(cur_person.vest_bbox, im0, label=f'v {cur_person.vest_iou:.2f}', color=colors[1], line_thickness=2)
                            
                        plot_skeleton_kpts(im0, skeleton_coords, 3)
                        # if save_crop:
                        #     txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                        #     save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

            print(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

        else:
            strongsort_list[i].increment_ages()
            print('No detections')

        # Stream results
        if show_vid:
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

        # Save results (image with detections)
        if save_vid:
            if vid_path[i] != save_path:  # new video
                vid_path[i] = save_path
                if isinstance(vid_writer[i], cv2.VideoWriter):
                    vid_writer[i].release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer[i].write(im0)

        prev_frames[i] = curr_frames[i]

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, imgsz, imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_pose_weights)  # update pose_model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-pose-weights', nargs='+', type=str, default=WEIGHTS / 'yolov7-w6-pose.pt', help='model.pt path(s)')
    parser.add_argument('--yolo-ppe-weights', nargs='+', type=str, default=WEIGHTS / 'best.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_5_market1501.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    # parser.add_argument('--max-pose_det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    # parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
