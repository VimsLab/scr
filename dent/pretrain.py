import os
import gc
import math
import torch
import pickle
import random
import argparse
import numpy as np
import torch.nn as nn
import concurrent.futures
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms

from PIL import Image
from glob import glob
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from torch.nn.parallel import DataParallel
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from model.transformer import Encoder
from utils.util import (get_pickles, ids, fold_operation, split, load_one_pickle)

torch.manual_seed(3500)
np.random.seed(3500)
random.seed(3500)

TQDM_BAR_FORMAT = '{desc} {n_fmt}/{total_fmt} [{elapsed} | {remaining} | {rate_fmt}]' #'{l_bar}{r_bar}' #'{l_bar}{r_bar}' # tqdm bar format
SAVE_PATH = 'runs/'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

from utils.plots import plot_images

def setup(rank, world_size):
    # Initialize the process group
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:28456",
        rank=rank,
        world_size=world_size
    )
    # Set the GPU to use
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def printer(vals, names):
    print('\n')
    for val, name in zip(vals, names):
        print(f'{name}: {val.shape}')


class SiameseDataset(Dataset):
    def __init__(self, rank, data, phase, transform=None, percent=1.0, num_workers=8, task='train'):
        self.phase = phase
        self.transform = transform
        self.data = data
        self.task = task
        


    def __getitem__(self, index):
        file = self.data[index]

        im_a = load_one_pickle(file[0])['img']
        im_b = load_one_pickle(file[1])['img']

        target = torch.ones((1))*file[2]

        assert im_a is not None 
        assert im_b is not None
        assert target is not None

        if self.transform:
            im_a = self.transform(im_a)
            im_b = self.transform(im_b)            

        if self.task =='infer':
            return im_a, im_b, target.squeeze(-1), file[0], file[1]

        else:
            return im_a, im_b, target.squeeze(-1)



    def __len__(self):
        return len(self.data)




class SiameseNetwork(nn.Module):
    def __init__(self, encoder):
        super(SiameseNetwork, self).__init__()
        self.encoder = encoder
        self.act = nn.Tanh()  
        
    def forward(self, x1, x2):
        embedding1, atn1 = self.encoder(x1)
        embedding2, atn2 = self.encoder(x2)

        return self.act(embedding1), self.act(embedding2), atn1, atn2


def get_distance(a,b):
    distance = nn.CosineSimilarity(dim=1, eps=1e-5)(a, b)
    mi, ma = -1, 1
    distance = (distance-mi)/(ma-mi)
    return 1-distance



def contrastive_focal_loss(rank, emb1, emb2, target, gamma=2, eps=1e-5, alpha=0.5):    
    
    distance = get_distance(emb1, emb2)
    p_t = torch.where(target==1, eps+1-distance, eps+distance)
    alpha = torch.ones_like(target)*alpha
    alpha_t = torch.where(target==1, alpha, 1-alpha)
    ce = -torch.log(p_t)
    cf_loss = ce * alpha_t * (1-p_t)**gamma

    # print([(d.item(), t.item(), c.item()) for d,t,c in zip(distance, target, cf_loss)])
  
    loss = torch.mean(cf_loss)   
    return loss


def save_model(root, siamese_net, epoch, optimizer, acc, best_accuracy, fold):
    if acc>=best_accuracy:
        best_accuracy = acc
        name = 'best_pretrainer.pth'
    else:
        name = 'last_pretrainer.pth'
    
    save_path = root + SAVE_PATH + str(fold) + name
    
    checkpoint = {
            'epoch': epoch,
            'model_state_dict': siamese_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_accuracy,
        }
    torch.save(checkpoint, save_path)
    print(('\n'+'%44s'+'%22s') %('Model saved successfully in ', save_path))
    return best_accuracy



def train_epoch(rank, siamese_net, fold, optimizer, train_loader, val_loader, best_accuracy, epoch, epochs, opt, running_loss=0):  

    if rank ==0:
        print(('\n' + '%44s'+'%22s' * 3) % ('Fold', 'Epoch', 'GPU Mem','Loss'))

    pbar = tqdm(enumerate(train_loader), bar_format=TQDM_BAR_FORMAT, total=len(train_loader))
    # with torch.autograd.detect_anomaly():
    for batch_idx, (x1, x2, targets) in pbar:
        x1 = x1.to(rank, non_blocking=True)
        x2 = x2.to(rank, non_blocking=True)
        targets = targets.to(rank, non_blocking=True)

        optimizer.zero_grad()
        
        # Forward pass
        with autocast():
            embeddings1, embeddings2, _, _ = siamese_net(x1, x2)
            loss = contrastive_focal_loss(rank, embeddings1[:,-1], embeddings2[:,-1], targets)

        # Backward pass and optimization
        loss.backward()

        optimizer.step()
        running_loss = running_loss+loss.item()

        if rank==0:
            # dist= get_distance(embeddings1[:,-1], embeddings2[:,-1])
            # print([(d.item(),t.item()) for d,t in zip(dist[:20], targets[:20])])
            
            mem = f'{torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%44s'+'%22s'*2 + '%22.4g') % 
                (f'{fold}', f'{epoch}/{epochs - 1}', mem, running_loss/(batch_idx+1)))

        # if rank==0 and batch_idx>20:
        #     break
    
    acc =  validate(rank, siamese_net, val_loader)
    if rank==0:
        best_accuracy=save_model(opt.root, siamese_net, epoch, optimizer, acc, best_accuracy, fold)

    # return siamese_net, optimizer
    return best_accuracy



def validate(rank, siamese_net, val_loader, thres=0.41):
    torch.cuda.empty_cache()
    gc.collect()

    siamese_net.train(False)
    with torch.no_grad():
        total_loss = 0 
        corrects = 0
        tps = 0
        tns = 0
        total = 0

        if rank==0:
            print(('\n' + '%44s'+'%22s' * 4) % ('Correct', '(TP,P)', '(TN,N)', 'Accuracy', 'Loss'))
        pbar = tqdm(enumerate(val_loader), bar_format=TQDM_BAR_FORMAT,total=len(val_loader))

        for batch_idx, (x1, x2, targets) in pbar:
            x1 = x1.to(rank, non_blocking=True)
            x2 = x2.to(rank, non_blocking=True)
            targets = targets.to(rank, non_blocking=True)

            # Forward pass
            embeddings1, embeddings2, _, _ = siamese_net(x1, x2)

            loss = contrastive_focal_loss(rank, embeddings1[:,-1], embeddings2[:,-1], targets)
            dist= get_distance(embeddings1[:,-1], embeddings2[:,-1])
            
            threshold = torch.ones_like(dist)*thres
            op = torch.relu(torch.sign(threshold-dist))


            # if rank==0:
            #     print([(d.item(),o.item(),t.item()) for d,o,t in zip(dist[:20], op[:20], targets[:20])])

          
            correct = op.eq(targets)
            tp = correct[op==1].sum().item()
            tn = correct[op==0].sum().item()

            p = targets.sum().item()
            n = len(targets) - p

            correct = correct.sum().item()
            tps += tp
            tns += tn
            total += targets.size(0)
            corrects += correct 

            # accumulate loss.item()
            total_loss += loss.item()

            if rank==0:
                pbar.set_description(('%44s'+'%22s'*2 +'%22.4g' * 2) % (correct, f'({tp},{p})', f'({tn},{n})', correct/(p+n), loss.item()))

            # if rank==0 and batch_idx>64:
            #     break
            

        # calculate average loss and accuracy
        avg_loss = total_loss / len(val_loader)
        accuracy = corrects / total
    if rank==0:
        print(('\n'+ '%44s') % ('Validation stats:'))
        print(('%44s'+'%22s' * 5) % ('Total', 'TP', 'TN', 'Incorrect', 'avg_acc', 'avg_loss'))
        print(('%44s'+'%22s' * 3 + "%22.4g"*2) % (total, f'{tps}/{corrects}', f'{tns}/{corrects}', total-corrects, accuracy, avg_loss))

    return accuracy


def tx():
    tx_dict = {'train':transforms.Compose([
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),

        'val': transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        }
    return tx_dict



def get_dataset(world_size, rank, data, phase, transform, batch_size=64, shuffle=False, num_workers=8, task='train'):

    dataset = SiameseDataset(rank, data, phase, task=task)
    if world_size>0:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    else:
        sampler=None

    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return dataloader, sampler



def pretrainer(rank, world_size, opt):
    num_epochs = opt.epochs
    batch_size = opt.batch_size 
    root = opt.root 
    phases = [opt.train_folder, opt.val_folder]
    resume = opt.resume
    resume_weight = opt.resume_weight
    folds = opt.folds
    fold = opt.cf


    setup(rank, world_size)
    
    tx_dict = tx()

    # while fold<folds:   
    # create model and optimizer
    encoder = Encoder(hidden_dim=256, num_encoder_layers=6, nheads=8)
    siamese_net = SiameseNetwork(encoder).to(rank)

    # Wrap the model with DistributedDataParallel
    siamese_net = DDP(siamese_net, device_ids=[rank], find_unused_parameters=False)
    optimizer = torch.optim.Adam(siamese_net.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_accuracy = 0
    start_epoch = 0

    if resume:
        ckptfile = root + SAVE_PATH + str(fold) +  resume_weight + '.pth'
        ckpts = torch.load(ckptfile, map_location='cpu')
        siamese_net.load_state_dict(ckpts['model_state_dict'])
        optimizer.load_state_dict(ckpts['optimizer_state_dict'])
        start_epoch = ckpts['epoch']+1
        best_accuracy = ckpts['best_val_acc']

        if rank == 0:
            print('\nResuming training from epoch {}. Loaded weights from {}. Last best accuracy was {}'
                .format(start_epoch, ckptfile, best_accuracy))


    # Train the network

    train, val = split(folds, fold)
    train_loader, train_sampler = get_dataset(world_size, rank, train,
                                            phase=phases[0], 
                                            transform=tx_dict['train'], 
                                            batch_size=batch_size)
    val_loader, val_sampler = get_dataset(world_size, rank, val,
                                        phase=phases[1],transform=tx_dict['val'], 
                                        batch_size=int(batch_size/4))

    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)
        best_accuracy = train_epoch(
                    rank, siamese_net, fold, optimizer, train_loader, 
                    val_loader, best_accuracy,
                    epoch, num_epochs, opt, running_loss=0
                    )
        lr_scheduler.step() 
        torch.cuda.empty_cache()
        gc.collect()
            
                
        # fold = fold+1
        # if fold<folds:

    del train_loader
    del val_loader
    del siamese_net
    del optimizer
    del lr_scheduler
    torch.cuda.empty_cache()
    gc.collect()

    cleanup()
      


__all__ = ['pretrainer', 'train_epoch', 'SiameseDataset', 'SiameseNetwork', 'contrastive_focal_loss', 'get_distance',
           'validate', 'tx', 'get_dataset', 'setup', 'cleanup']



def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/media/jakep/eye/scr/dent/', help='project root path')
    parser.add_argument('--world_size', type=int, default=2, help='World size')
    parser.add_argument('--resume', type=bool, default=False, help='To resume or not to resume')
    parser.add_argument('--resume_weight', type=str, default='best_pretrainer', help='path to trained weights if resume')
    parser.add_argument('--train_folder', type=str, default='train2', help='name of the directory containing training samples')
    parser.add_argument('--val_folder', type=str, default='val2', help='name of the directory containing validation samples')    
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--folds', type=int, default=5, help='number of dataset folds for training')
    parser.add_argument('--cf', type=int, default=0, help='fold number to train. Must be provided if resume is not False')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')

    return parser.parse_args()



if __name__ == '__main__':

    iterate = False
    opt = arg_parse()

    mp.spawn(pretrainer, args=(opt.world_size, opt), nprocs=opt.world_size, join=True)
        


