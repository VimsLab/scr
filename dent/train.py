import os
import re
import gc
import pdb
import torch
import pickle
import random
import argparse
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms

from PIL import Image
from glob import glob
from tqdm import tqdm
from pathlib import Path
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from torchvision.ops import complete_box_iou_loss
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

from pretrain import *
from utils.plots import plot_images
from utils.dl import create_dataloader
from matching_loss import build_matcher
from utils.general import xywh2xyxy, xyxy2xywh
from loss_criterion import *
# from validate import epoch_validate
from transformer import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
# os.environ['NCCL_DEBUG']='INFO'




def setup(rank, world_size):
	# Initialize the process group
	dist.init_process_group(
		backend="nccl",
		init_method="tcp://127.0.0.1:12426",
		rank=rank,
		world_size=world_size,
		timeout=datetime.timedelta(seconds=5000)
	)
	# Set the GPU to use
	torch.cuda.set_device(rank)


def cleanup():

	dist.destroy_process_group()


def get_loader(dataset, batch_size):
	sampler = DistributedSampler(dataset, shuffle=True)
	data_loader = DataLoader(dataset,
				  batch_size=batch_size,
				  shuffle=False,
				  num_workers=6,
				  sampler=sampler, 
				  drop_last=False,              
				  collate_fn=dataset.collate_fn)
	return data_loader




def get_dataset(rank, world_size,dataroot, phase, batch_size, r, space):
	dataset = create_dataloader(dataroot+phase,
								576,
								batch_size, 
								rank=rank,                                   
								cache='ram', # if opt.cache == 'val' else opt.cache,
								workers=6,
								phase=phase,
								shuffle=True,
								r=r,
								space=space)
	
	return dataset



def compute_loss(outputs, targets, criterion, nc=2):

	loss_dict = criterion(outputs, targets) #.to(outputs.device)

	weight_dict = criterion.weight_dict
	losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

	return losses


def train_epoch(rank, model, optimizer, train_loader, epoch, epochs, criterion, nc, running_loss=0):
	model.train()
	gc.collect()

	criterion.train()

	if rank==0:
		print(('\n' + '%22s' * 4) % ('Device', 'Epoch', 'GPU Mem', 'Loss'))

	pbar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:+10b}')

	for batch_idx, (img, targets) in pbar:
		img = img.to(rank, non_blocking=True)
		targets = [t.to(rank, non_blocking=True) for t in targets]
		optimizer.zero_grad()


		tgt = []
		for i,t in enumerate(targets):
			tg = torch.zeros((len(t), 6))
			tg[:,1:] = t 
			tg[:,0] = i
			tgt.append(tg)

		tgt = torch.cat(tgt)

		# if rank==0:
		# 	print(torch.max(img[:,1]), torch.min(img[:,1]))
		# 	plot_images(img[:,1].cpu(), tgt.cpu(), fname=str(batch_idx)+'.png')
		# Forward pass
		outputs = model(img.permute(1,0,2,3,4))
		outputs = {'pred_logits':outputs[0], 'pred_boxes': outputs[1]}
		targets = [{'labels': t[:,0], 'boxes':t[:,1:]} for t in targets]

		# loss = compute_loss(outputs, targets[:,1:], phase='train', objthres=0.001)
		loss = compute_loss(outputs, targets, criterion)

		# Backward pass and optimization
		loss.backward()
		optimizer.step()

		running_loss = running_loss+loss.item()

		if rank==0:
			mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
			pbar.set_description(('%22s'*3 + '%22.4g' * 1) % (f'{torch.cuda.current_device()}', f'{epoch}/{epochs - 1}', mem, running_loss/(batch_idx+1)))
			
	# print(A)
	return model


# def detector(rank, world_size, root, dataroot, pretraining=False, pretrained_weights_path='best_pretrainer.pth', resume=False):
def detector(rank, world_size, opt):
	setup(rank, world_size)
	# trainig params
	# nc = 2
	# epochs = 152
	# r = 3
	# space = 1
	# batch_size = 16
	# val_batch_size = 16

	nc = opt.nc
	epochs = opt.epochs
	r = opt.r
	space = opt.space
	batch_size = opt.train_batch
	val_batch_size = opt.val_batch

	dataroot = opt.dataroot
	root = opt.root

	pretraining = opt.pretrain 
	pretrained_weights_path = opt.pretrain_weights

	resume = opt.resume

	# if rank>-1:
	train_data = get_dataset(rank, world_size, dataroot, 'val2', batch_size, r, space)
	val_dataset = get_dataset(rank, world_size, dataroot, 'val2', batch_size, r, space)
	gc.collect()
	
	# define detection model
	model = Dent(hidden_dim=256, num_class=2).to(rank)
	model = DDP(model, device_ids=[rank], find_unused_parameters=True)

	# declare optimizer and scheduler
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

	train_loader = get_loader(train_data, batch_size)
	val_loader = get_loader(val_dataset, val_batch_size)

	best_fitness = 0
	del train_data
	torch.cuda.empty_cache()
	gc.collect()

	criterion_train,_ = loss_functions(nc, phase='train')
	criterion_val,_ = loss_functions(nc, phase='val')

	start_epoch = 0

	if resume:
		ckptfile = root + resume + '.pth'
		ckpts = torch.load(ckptfile, map_location='cpu')
		model.load_state_dict(ckpts['model_state_dict'])
		optimizer.load_state_dict(ckpts['optimizer_state_dict'])
		start_epoch = ckpts['epoch']+1
		best_accuracy = ckpts['best_val_acc']

		if rank == 0:
			print('Resuming training from epoch {}. Loaded weights from {}. Last best accuracy was {}'
				.format(start_epoch, ckptfile, best_accuracy))

	
	for epoch in range(start_epoch, epochs):
		model = train_epoch(rank, model, optimizer, train_loader, epoch, epochs, criterion_train, nc)
		
		# Update the learning rate
		lr_scheduler.step()

		epoch_validate(rank, model, val_loader, criterion_val, nc)
		
		save_path = f'{root}outputs/detection_{epoch}.pth'
		if rank==0:
			checkpoint = {
					'epoch': epoch,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'best_val_acc': best_fitness,
				}
			torch.save(checkpoint, save_path)
	cleanup()
		


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/media/jakep/eye/scr/dent/', help='project root path')
    parser.add_argument('--dataroot', type=str, default='/media/jakep/eye/scr/pickle/', help='path to pickled dataset')
    parser.add_argument('--world_size', type=int, default=2, help='World size')
    parser.add_argument('--resume', type=str, default='False', help='path to trained weights or "False"')
    parser.add_argument('--pretrain', type=bool, default=True, help='Begin with pretrained weights or not. Must provide path if yes.')
    parser.add_argument('--pretrain_weights', type=str, default='best_pretrainer.pth', help='path to pretrained weights. --pretrain must be true')
    
    parser.add_argument('--nc', type=int, default=2, help='number of classes')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--r', type=int, default=3, help='number of adjacent images to stack')
    parser.add_argument('--space', type=int, default=1, help='Number of steps/ stride for next adjacent image block')
    parser.add_argument('--train_batch', type=int, default=64, help='training batch size')
    parser.add_argument('--val_batch', type=int, default=16, help='validation batch size')

    return parser.parse_args()



if __name__ == '__main__':

	opt = arg_parse()

	# root = '/media/jakep/eye/scr/dent/'
	# dataroot = '/media/jakep/eye/scr/pickle/'

	# # ddp
	# world_size = 2
	mp.spawn(detector, args=(opt.world_size, opt), nprocs=opt.world_size, join=True)