"""
Dataloaders and dataset utils
"""

import contextlib
import glob
import pickle
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torchvision
import yaml
from PIL import Image, ImageOps
from torchvision.ops import nms
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

# from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
#                                  cutout, letterbox, mixup, random_perspective)
from utils.general import (check_dataset, check_requirements, check_yaml, clean_str,
                           colorstr, cv2, unzip_file, xyn2xy, xywh2xyxy,
                           xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first


random.seed(888100)

# Parameters
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders



def create_dataloader(path,
                      batch_size,
                      rank,
                      augment=False,
                      cache=False,
                      workers=8,
                      phase='val',
                      shuffle=False,
                      r=3,
                      space=1):
    
    print('Image cache=', cache, 'Shuffle=', shuffle, "Data=", phase, "Rank=", rank)
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabels(
            path,
            batch_size,
            augment=augment,  
            cache_images=cache,
            phase=phase,
            r=r,
            space=space)
    return dataset


class LoadImagesAndLabels(Dataset):

    def __init__(self,
                 path,
                 batch_size=16,
                 augment=False,
                 cache_images=False,
                 phase='val',
                 r=3,
                 space=1):
        
        self.augment = augment
        self.path = path
        self.phase = phase
        self.r = r
        self.space = space


        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f'{p} does not exist')
            self.im_files = [x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in ['pkl']]
            random.shuffle(self.im_files)
            assert self.im_files, f'No images found'
        except Exception as e:
            raise Exception(f'Error loading data from {path}: {e}\n{HELP_URL}') from e


        n = len(self.im_files)  # number of images
        self.n = n
        self.indices = range(n)


        par = Path(self.im_files[0]).parent
        sar = par.name + 'filename'
        car = par.name + 'index'
        filecache_path = (par.parent/sar).with_suffix('.cache')
        indexcache_path = (par.parent/car).with_suffix('.cache')
        c_file, c_index = self.cache_files(filecache_path, indexcache_path)

        self.roots = c_file
        self.c_index = c_index

    def cache_files(self, p, q):
        if p.is_file() and q.is_file():
            files= np.load(p, allow_pickle=True).item()
            indexfile= np.load(q, allow_pickle=True).item()
            return files, indexfile

        files = {}
        roots = []
        indexfile = {}
        print('\nCreating index to root caches since they dont exist (yet)...\n')
        for i in range(len(self.im_files)):
            f = self.im_files[i]
            root, tail = self.get_roots(f)
            indexfile[f] = [i, root, tail]

            if root not in roots:
                roots.append(root)
                files[root] = [tail]
            else:
                files[root].append(tail)
                files[root] = sorted(files[root])

        np.save(p, files)
        p.with_suffix('.cache.npy').rename(p)

        np.save(q, indexfile)
        q.with_suffix('.cache.npy').rename(q)

        return files, indexfile


    def get_roots(self, file):
        name = file.split('_')
        ext = name[-1].split('.')[0]
        return name[0], int(ext)


    def __len__(self):
        return len(self.im_files)


        # *************************************************************************************************

    def create_labelout(self, b,c):
        # [bi, class, x, y, w, h]
        c = [cl+1 if cl <2 else 0 for cl in c]
        if 0 in c:
            b = torch.empty((0,4))
            c = torch.zeros((0,1))
        else:
            b = torch.tensor(b)
            c = torch.tensor(c).reshape((-1,1))

        d = torch.zeros_like(c)
        lab = torch.cat([d,c,b], dim=-1).float()
        return lab



    def lookup(self, index):
        if self.r == 1:
            return [index]
        file = self.im_files[index] # index filename
        i, root, tail = self.c_index[file] # filename [index, root, tail]
        a = np.array(self.roots[root]) # root [tail1, tail2, ....]
        n1 = np.where(a==tail)[0][0] # tail index

        n0 = n1 - self.space
        n2 = n1 + self.space

        neighborhood = [0,index,0]

        if n0<0:
            neighborhood[0] = index
        else:
            filename = root + '_' + str(a[n0]) + '.pkl'
            # print(filename)
            # n0_ind = np.where(self.im_files==filename)

            n0_ind = self.c_index[filename][0]
            neighborhood[0] = n0_ind

        if n2>=len(a):
            neighborhood[2] = index
        else:
            filename = root + '_' + str(a[n2]) + '.pkl'
            n2_ind = self.c_index[filename][0]
            neighborhood[2] = n2_ind

        return neighborhood




    def load_pickle(self, index):
        im_file = self.im_files[index]
        with open(im_file, 'rb') as handle:
            d = pickle.load(handle)
            im = d['img']
            bo = d['box']
            clss = d['class']
        return im, bo, clss



    def __getitem__(self,index):
        r=self.r
        space=self.space

        neighborhood = self.lookup(index)

        image = [None]*r
        labels = []
        for i, ind in enumerate(neighborhood):
            img, box, clss = self.load_pickle(ind)
            img = cv2.merge([img, img, img])
            image[i] = img
            if ind == index:
                labels = self.create_labelout(box, clss)
             
        image = np.stack(image)

        image = torch.from_numpy(np.ascontiguousarray(image).astype(np.float32)).permute(0,3,1,2)
        return image, labels


    @staticmethod
    def collate_fn(batch):
        im, label = zip(*batch)  # transposed        
       
        label = [l[:,1:] for l in label]
        im = torch.stack(im, 0)

        return im, label




