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
from collections import OrderedDict
from matplotlib import pyplot as plt
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from torchvision.ops import complete_box_iou_loss
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

from pretrain import *
from model.transformer import *
from utils.plots import plot_images
from utils.dataloader import create_dataloader
from loss.matching_loss import build_matcher
from utils.util import plot_attention, pltbbox, id_per_file, read_data
from utils.general import xywh2xyxy, xyxy2xywh
from loss.loss_criterion import *
# from train import get_loader, get_dataset					pltbbox(m.cpu(),p[:,
from loss.matching_loss import box_cxcywh_to_xyxy, box_iou
from eval.metrics import MetricLogger, SmoothedValue, accuracy
from eval.metrices import *
from eval.eval import EvaluationCriterion

torch.manual_seed(1213)
np.random.seed(2022)
random.seed(1027)

import wandb

wb=False

TQDM_BAR_FORMAT = '{desc} {n_fmt}/{total_fmt} [{elapsed} | {remaining} | {rate_fmt}]'



def compute_loss(outputs, targets, criterion):

	loss_dict = criterion(outputs, targets) #.to(outputs.device)

	weight_dict = criterion.weight_dict
	losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
	
	loss_reduced = reduce_dict(loss_dict)
	return losses, loss_reduced



def epoch_validate(rank, model, val_loader, criterion, nc, plot=True, wb=False):
	# torch.cuda.empty_cache()
	# gc.collect()

	criterion.eval()
	# model.eval()
	
	# postprocessors = {'bbox': PostProcess()}
	
	if rank==0:
		print(('\n' + '%44s'+'%22s' * 4) % ('***Validation***', 'bbox_loss', 'ce_loss', 'giou_loss', 'total_loss'))
	pbar = tqdm(enumerate(val_loader), total=len(val_loader), bar_format=TQDM_BAR_FORMAT)

	eval_criterion = EvaluationCriterion(rank, plot=plot)
	lossess = []


	for batch_idx, (images, targets, fn) in pbar:
		
		images = images.to(rank)
		targets = [t.to(rank, non_blocking=True) for t in targets]

		with torch.no_grad():

			# Forward pass
			outputs = model(images.permute(1,0,2,3,4))
			outputs = {'pred_logits':outputs[0], 'pred_boxes': outputs[1]}
			target = [{'labels': t[:,0], 'boxes':t[:,1:]} for t in targets]

			
			loss, batch_loss = compute_loss(outputs, target, criterion)
			if wb:
				wandb.log({'val_loss': loss})
				wandb.log({'val_'+a:b for a,b in batch_loss.items()})
			# plot_attention(images[1].detach().cpu(), atn.permute(0,2,1).detach().cpu(), fn)
			lossess.append(loss.item())

			# results = postprocessors['bbox'](outputs, images.shape[-2:])
			# res = {batch_idx: [output for output in results]}
			
			if rank==0:
				pbar.set_description(('%44s'+'%22.4g' * 4) % (f' ', 
														batch_loss['loss_bbox'].item(), 
														batch_loss['loss_ce'].item(), 
														batch_loss['loss_giou'].item(), 
														loss.item()))

			
			eval_criterion.evalcriterion(batch_idx, images[:,1], targets, outputs, fn, save_dir='valimages/')
			# if batch_idx>2:
			# 	break
	loss=sum(lossess)/len(lossess)	
	fitness = eval_criterion.calcmetric(plot=plot, save_dir='valimages/', loss=loss)
			
	return fitness, loss



def setup(rank, world_size):
	# Initialize the process group
	dist.init_process_group(
		backend="nccl",
		init_method="tcp://127.0.0.1:12426",
		rank=rank,
		world_size=world_size,
		timeout=datetime.timedelta(seconds=5)
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

def load_saved_model(weights_path, root, M, O=None):
	ckptfile = root + 'runs/' + weights_path + '.pth'
	ckpts = torch.load(ckptfile, map_location='cpu')
	ckpt = ckpts['model_state_dict']
	if O is None:
		new_state_dict = OrderedDict()
		for key, value in ckpt.items():
			new_key = key.replace('module.encoder.', '')
			new_state_dict[new_key] = value
		M.load_state_dict(new_state_dict)

	
	if O is not None:
		M.load_state_dict(ckpt)
		O.load_state_dict(ckpts['optimizer_state_dict'])
		start_epoch = ckpts['epoch']+1
		best_accuracy = ckpts['best_fitness']
		return M, O, start_epoch, best_accuracy

	return M

def validate(rank, world_size, opt):
	setup(rank, world_size)
	
	batch_size = opt.batch 
	nc = opt.nc
	r = opt.r 
	space = opt.space

	dataroot = opt.dataroot
	root = opt.root 

	weights = opt.weights

	data = read_data('val')
	val_dataset = create_dataloader(data,
								dataroot,
								batch_size, 
								rank=rank,                                   
								cache='ram', # if opt.cache == 'val' else opt.cache,
								workers=6,
								phase='val',
								shuffle=True,
								r=r,
								space=space)
	
	
	# get_dataset(rank, world_size, dataroot, 'val2', batch_size, r, space)
	
	# define detection model
	encoder = Encoder(hidden_dim=256,num_encoder_layers=6, nheads=8).to(rank)
	encoder = load_saved_model('0best_pretrainer', root, encoder, None)
	model = Dent_Pt(encoder, hidden_dim=256, num_class=2).to(rank)
	model = DDP(model, device_ids=[rank], find_unused_parameters=True)

	val_loader = get_loader(val_dataset, batch_size)
	if wb:
		wandb.init(
		project="scr", 
		name="test", 
		config={
		"architecture": "DENT",
		"dataset": "SCR",
		"steps": len(val_loader),
		"batch":32,
		"num_classes": 2, 
		"class_names": ['Fovea, SCR']
		
		})
  

	ckptfile = root + weights + '.pth'
	ckpts = torch.load(ckptfile)
	model.load_state_dict(ckpts['model_state_dict'])
	# model = torch.load(ckptfile)
	# best_accuracy = ckpts['best_fitness']

	criterion_val,_ = loss_functions(nc, phase='val')
	# rank = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	epoch_validate(rank, model, val_loader, criterion_val, nc, wb=wb)
	if wb:
		wandb.finish()
	cleanup()




def arg_parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--root', type=str, default='/media/jakep/eye/scr/dent/', help='project root path')
	parser.add_argument('--dataroot', type=str, default='/media/jakep/eye/scr/dent/data', help='path to pickled dataset')
	parser.add_argument('--world_size', type=int, default=1, help='World size')
	parser.add_argument('--weights', type=str, default='outputs/detection_best', help='path to trained weights')
	
	parser.add_argument('--nc', type=int, default=2, help='number of classes')
	parser.add_argument('--r', type=int, default=3, help='number of adjacent images to stack')
	parser.add_argument('--space', type=int, default=1, help='Number of steps/ stride for next adjacent image block')
	parser.add_argument('--batch', type=int, default=32, help='validation batch size')

	return parser.parse_args()



if __name__ == '__main__':
	if wb:
		wandb.login()
	opt = arg_parse()
	mp.spawn(validate, args=(opt.world_size, opt), nprocs=opt.world_size, join=True)