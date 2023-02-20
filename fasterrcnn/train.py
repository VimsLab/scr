# python train.py --img 576 --batch 4 --epoch 2 --cache --device cpu --r 1 --space 1

import os
import sys
import yaml
import math
import time
import torch
import random
import argparse
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.distributed as dist
import torchvision.transforms.functional as F

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from torch.optim import lr_scheduler
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

import val as validate
from utils.loggers import Loggers
from utils.metrics import fitness
from utils.callbacks import Callbacks
from utils.dataloader import create_dataloader
from utils.plots import plot_results, plot_images
from utils.downloads import attempt_download, is_url
from utils.torch_utils import (EarlyStopping, smart_load, de_parallel, smart_DDP, smart_optimizer, smart_resume, select_device, torch_distributed_zero_first)
from utils.general import (LOGGER, methods, strip_optimizer, colorstr, xyxy2xywh, TQDM_BAR_FORMAT, Profile, increment_path, init_seeds, yaml_save, check_dataset)

import warnings
warnings.filterwarnings('ignore')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))



# def plot_results(file='path/to/results.csv', dir=''):
#     # Plot training results.csv. Usage: from utils.plots import *; plot_results('path/to/results.csv')
#     save_dir = Path(file).parent if file else Path(dir)
#     fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
#     ax = ax.ravel()
#     files = list(save_dir.glob('results*.csv'))
#     assert len(files), f'No results.csv files found in {save_dir.resolve()}, nothing to plot.'
#     for f in files:
#         data = pd.read_csv(f)
#         s = [x.strip() for x in data.columns]
#         x = data.values[:, 0]
#         for i, j in enumerate(range(10)):
#             y = data.values[:, j].astype('float')
#             # y[y == 0] = np.nan  # don't show zero values
#             ax[i].plot(x, y, marker='.', label=f.stem, linewidth=2, markersize=8)
#             ax[i].set_title(s[j], fontsize=12)
#             # if j in [8, 9, 10]:  # share train and val loss y axes
#             #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        
#     ax[1].legend()
#     fig.savefig(save_dir / 'results.png', dpi=200)
#     plt.close()

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

def save_csv(save_dir, keys, vals, epoch):
    x = dict(zip(keys, vals))
    file = save_dir / 'results.csv'
    n = len(x) + 1  # number of cols
    s = '' if file.exists() else (('%20s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # add header
    with open(file, 'a') as f:
        f.write(s + ('%20.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

def train(opt, callbacks): 

    save_dir, epochs, batch_size, pretrained, data, cfg, resume, noval, nosave, workers, freeze, imgsz = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.pretrained, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze, opt.imgsz
    
    callbacks.run('on_pretrain_routine_start')

    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # save run settings
    yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Pretrained
    weights = None
    hyp = None
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT


    # Loggers
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset

        if resume:  # If resuming runs from remote artifact
            pretrained, epochs, batch_size = opt.pretrained, opt.epochs, opt.batch_size


    device = select_device(opt.device, batch_size=opt.batch_size)
    
    init_seeds(opt.seed + 1 + RANK, deterministic=True)

    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None

    # YAML data
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = int(data_dict['nc'])  # number of classes
    names = data_dict['names'] 
    lr = data_dict['lr']

    cuda = device.type != 'cpu'

    # Model
    model = fasterrcnn_resnet50_fpn(
        weights=weights, 
        progress=True, 
        num_classes=nc,
        rpn_fg_iou_thresh=0.2,
        rpn_bg_iou_thresh=0.05, 
        max_size=428,
        )
    model=model.to(device)


    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr)

    # scheduler
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  #
    # plot_lr_scheduler(optimizer, scheduler, epochs)


    # Resume
    best_fitness, start_epoch = 0.0, 0
    if resume is not None and resume.endswith('.pt'):
        weights = resume
        ckpt = torch.load(weights, map_location='cpu')
        best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, epochs)
        del ckpt    

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # Trainloader
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size,
                                              augment=False,
                                              cache=opt.cache, # if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              phase='train',
                                              shuffle=True,
                                              r=opt.r,
                                              space=opt.space)

    labels = dataset.lo

    # Valloader
    if RANK in {-1, 0}:
        val_loader = create_dataloader(val_path,
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

        callbacks.run('on_pretrain_routine_end', labels, names)

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)


    t0 = time.time()
    nb = len(train_loader)
    maps = np.zeros(nc)
    results = (0, 0, 0, 0)
    # scheduler.last_epoch = start_epoch - 1  # do not move

    stopper, stop = EarlyStopping(patience=opt.patience), False

    callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')

    for epoch in range(start_epoch, epochs):  
        callbacks.run('on_train_epoch_start')
        model.train()
        model=model.double()

        mloss = torch.zeros(4, device=device)

        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)

        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 8) % ('Epoch', 'GPU_mem', 'cls_loss', 'box_loss', 'obj_loss', 'rpn_loss','Instances', 'Size'))

        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)
        
        optimizer.zero_grad()
        for i, (images, targets) in pbar:

            callbacks.run('on_train_batch_start')
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()

            loss_dict = model(images, targets) # loss in dict form
            losses = sum(loss for loss in loss_dict.values()) # sum of all losses
            loss_items = torch.stack(list(loss_dict.values())) # values of loss_dict
            
            losses.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients

            optimizer.step()
            

            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 6) %
                                         (f'{epoch}/{epochs - 1}', mem, *mloss.cpu().detach().numpy(), targets[0]['boxes'].shape[0], images[0].shape[-1]))

                #*****************for image plot**************
                tt = None
                if i < 3:
                    f = save_dir / f'train_batch{i}.jpg'  # filename
                    kx = [len(t['boxes']) for t in targets]
                    x = sum(kx) # total number of boxes in all images in the batch
                    tt = torch.zeros((x, 6))

                    ara = torch.arange(len(targets))
                    aran = [[a]*k for a,k in zip(ara, kx)]
                    aran = [a for j in aran for a in j]

                    lb = [a['labels'] for a in targets]
                    lb = [a for j in lb for a in j]

                    bx = [xyxy2xywh(a['boxes']) for a in targets]
                    bx = [a for j in bx for a in j]
                    

                    tt[:,0] = torch.stack(aran) # image index of the batch 
                    tt[:,1] = torch.stack(lb) # labels of each image
                    tt[:,2:] = torch.stack(bx) # boxes of each image

                    plot_images(torch.stack(images), tt, paths=None, fname=f)

                callbacks.run('on_train_batch_end', model, i, torch.stack(images), tt, paths=None, vals=list(mloss))
                #*****************for image plot**************

                if callbacks.stop_training:
                    return

            # XXXXXXXXXXXXXXXXXXX End batch XXXXXXXXXXXXXXXXXXXXXXXXX

        # Run validation
        if RANK in {-1, 0}:
            print('\n' + 'Validation')
            callbacks.run('on_train_epoch_end', epoch=epoch)
            results = validate.run(
                data=data_dict,
                batch_size=opt.batch_size,
                imgsz=imgsz,
                model=model,
                iou_thres=0.05,
                conf_thres=0.001,
                dataloader=val_loader,
                callbacks=callbacks,
                save_dir=save_dir,
                )

            fi = fitness(np.array(results).reshape(1, -1))
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                    best_fitness = fi

            log_vals = list(mloss) + list(results) + [lr]
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)
            # save_csv(save_dir, keys, log_vals, epoch)

            # Save model
            ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'fitness': fi,
                    'model': deepcopy(de_parallel(model)),
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'date': datetime.now().isoformat()
                    }

            # Save last, best and delete
            torch.save(ckpt, last)
            if best_fitness == fi:
                torch.save(ckpt, best)
            del ckpt
            callbacks.run('on_model_save', last, epoch, best_fitness, fi)

        if stop:
            break
        plot_results(file=save_dir / 'results.csv')
        # XXXXXXXXXXXXXXXXX End epoch XXXXXXXXXXXXXXXXXXXXXXXXXXX
     
    # XXXXXXXXXXXXXXXXXXX End training XXXXXXXXXXXXXXXXXXXXXX     
    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results = validate.run(
                        data_dict,
                        batch_size=opt.batch_size,
                        imgsz=imgsz,
                        model=smart_load(model, f, inf=True),
                        iou_thres=0.20,  
                        dataloader=val_loader,
                        save_dir=save_dir,
                        callbacks=callbacks,
                        )  # val best model with plots
                    
        callbacks.run('on_train_end', last, best, epoch, results)      
    torch.cuda.empty_cache()
    


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', action='store_true', help='initiate with pretrained weights')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/scr.yaml', help='dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', type=str, default=None, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='Adam', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=150, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--r', type=int, default=1, help='Number of frames to process at once')
    parser.add_argument('--space', type=int, default=1, help='How many frames to skip while processing r frames')

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):   
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)) 
    train(opt, callbacks)


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
