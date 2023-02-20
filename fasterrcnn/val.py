# python train.py --img 576 --batch 4 --epoch 2 --cache --device cpu --r 1 --space 1


import argparse
import math
import os
import random
import sys
import time
import yaml
import torch
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.distributed as dist
import torchvision.transforms.functional as F

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm
from torch.optim import lr_scheduler
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from utils.dataloader import create_dataloader
from utils.callbacks import Callbacks
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.torch_utils import select_device, smart_inference_mode, smart_load
from utils.plots import plot_results, plot_images, output_to_target
from utils.general import (LOGGER, TQDM_BAR_FORMAT, xyxy2xywh, Profile, increment_path, init_seeds, yaml_save, check_dataset)


import warnings
warnings.filterwarnings('ignore')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()



def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = (labels[:, 0:1] == detections[:, 5])
    iouv = iouv.to('cpu')
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
        data,
        weights=None,
        batch_size=1,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.05,  # NMS IoU threshold
        max_det=15,  # maximum detections per image
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        verbose=False,  # verbose output
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        compute_loss=None,
        callbacks=Callbacks(),
        phase='val',
        r=3,
        space=1
):

    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device
    
    else:  # called directly
        device = select_device(device, batch_size=batch_size)
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            progress=True,
            rpn_score_thresh=conf_thres,
            box_score_thresh=iou_thres,
            box_nms_thresh=0.5,
            box_detections_per_img=max_det,
            )

        model = smart_load(model, weights, inf=True)

    model.eval()

    cuda = device.type != 'cpu'

    data_dict = check_dataset(data)  # check
    val_path = data_dict[phase]
    nc = int(data_dict['nc'])  # number of classes
    names = data_dict['names'] 

    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel() # 10


    if not training:
        dataloader = create_dataloader(val_path,
                           imgsz,
                           batch_size,
                           cache=opt.cache, #if noval else opt.cache,
                           rect=True,
                           rank=-1,
                           workers=workers * 2,
                           pad=0.5,
                           phase='val',
                           r=opt.r,
                           space=opt.space)[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile()  # profiling times
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)
    stats, ap, ap_class = [], [], []
    callbacks.run('on_val_start')

    for i, (images, targets) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        with dt[0]:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with dt[1]:
            model = model.double()
            preds = model(images)
        
        # for result of each image in the batch
        for si, detect in enumerate(preds): 
            target = targets[si]  
            labels = torch.zeros((len(target['labels']), 5))
            labels[:,0] = target['labels']
            labels[:,1:] = target['boxes']

            pred = torch.zeros((len(detect['labels']), 6))
            pred[:,:4] = detect['boxes']
            pred[:, 4] = detect['scores']
            pred[:, 5] = detect['labels']


            nl, npr = labels.shape[0], pred.shape[0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            seen += 1

            # if no predictions were made
            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2,0), device=device), labels[:, 0]))
                    confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            if nl:
                predn = pred.clone()
                labelsn = labels.clone()
                correct = process_batch(predn, labelsn, iouv)
                confusion_matrix.process_batch(predn, labelsn)

            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))
            callbacks.run('on_val_image_end', pred, None, None, names, images[si])

        #**********************Plot images*******************************
        if i < 10:
            kx = [0,0]
            kx[0] = [len(t['boxes']) for t in targets]
            x = sum(kx[0]) # total number of boxes in all images in the batch
            tt = torch.zeros((x, 6))

            kx[1] = [len(t['boxes']) for t in preds]
            x = sum(kx[1]) # total number of boxes in all images in the batch
            pp = torch.zeros((x, 7))

            for ival, tsr in enumerate([targets, preds]):
                ara = torch.arange(len(tsr))
                aran = [[a]*k for a,k in zip(ara, kx[ival])]
                aran = [a for j in aran for a in j]

                lb = [a['labels'] for a in tsr]
                lb = [a for j in lb for a in j]

                bx = [xyxy2xywh(a['boxes']) for a in tsr]
                bx = [a for j in bx for a in j]
            
                if ival==0:
                    tt[:,0] = torch.stack(aran) # image index of the batch 
                    tt[:,1] = torch.stack(lb) # labels of each image
                    tt[:,2:] = torch.stack(bx) # boxes of each image

                if ival==1:
                    scx = [a['scores'] for a in tsr]
                    scx = [a for j in scx for a in j]

                    pp[:,0] = torch.stack(aran) # image index of the batch 
                    pp[:,2:6] = torch.stack(bx)  # boxes of each image
                    pp[:,6] = torch.stack(scx)  # scores of each box in each image
                    pp[:,1] = torch.stack(lb)   # labels of each box in each image
                del tsr

            plot_images(torch.stack(images), tt, fname=save_dir / f'val_batch{i}_labels.jpg', names=names)  # labels
            plot_images(torch.stack(images), pp, fname=save_dir / f'val_batch{i}_pred.jpg', names=names)  # pred
        callbacks.run('on_val_batch_end', i, images, targets, None, None, preds)
        
        #*****************************************************************    


    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=True, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
    
    # print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format    
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # print results per class
    if len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
        
    # print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: Pre-process: %.1fms, Inference: %.1fms\n' % t)

    # Plot
    confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    callbacks.run('on_val_end', nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    # Return results
    model.float()
    return (mp, mr, map50, map)



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/scr.yaml', help='dataset.yaml path')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--r', type=int, default=1, help='Number of frames to process at once')
    parser.add_argument('--space', type=int, default=1, help='How many frames to skip while processing r frames')

   
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    print_args(vars(opt))
    return opt


def main(opt):   
    run(**vars(opt))


# def run(**kwargs):
#     # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
#     opt = parse_opt(True)
#     for k, v in kwargs.items():
#         setattr(opt, k, v)
#     main(opt)
#     return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
