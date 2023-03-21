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
from matplotlib import pyplot as plt
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from torchvision.ops import complete_box_iou_loss
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

from pretrain import *
from transformer import *
from utils.plots import plot_images
from utils.dl import create_dataloader
from matching_loss import build_matcher
from utils.general import xywh2xyxy, xyxy2xywh
from loss_criterion import *
from train import get_loader, get_dataset
from matching_loss import box_cxcywh_to_xyxy
from metrics import MetricLogger, SmoothedValue, accuracy


class PostProcess(nn.Module):
	""" This module converts the model's output into the format expected by the coco api"""
	@torch.no_grad()
	def forward(self, outputs):
		""" Perform the computation
		Parameters:
			outputs: raw outputs of the model
			target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
						  For evaluation, this must be the original image size (before any data augmentation)
						  For visualization, this should be the image size after data augment, but before padding
		"""
		out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

		# assert len(out_logits) == len(target_sizes)
		# assert target_sizes.shape[1] == 2

		prob = F.softmax(out_logits, -1)
		scores, labels = prob[..., :-1].max(-1)

		# convert to [x0, y0, x1, y1] format
		boxes = box_cxcywh_to_xyxy(out_bbox)
		# and from relative [0, 1] to absolute [0, height] coordinates
		img_h, img_w = (224,256)
		# scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
		# boxes = boxes * scale_fct[:, None, :]
		# print(boxes.shape, boxes)
		boxes[:,:,::2] *= img_w
		boxes[:,:,1::2] *= img_h

		results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

		return results


def compute_loss(outputs, targets, criterion, nc=2):

	loss_dict = criterion(outputs, targets) #.to(outputs.device)

	weight_dict = criterion.weight_dict
	losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
	
	loss_reduced = reduce_dict(loss_dict)
	return losses, loss_reduced



def epoch_validate(rank, model, val_loader, criterion, nc):
	model.eval()
	criterion.eval()
	postprocessors = {'bbox': PostProcess()}
	

	with torch.no_grad():
		if rank==0:
			print(('\n' + '%22s' * 5) % ('cardn_loss', 'bbox_loss', 'ce_loss', 'giou_loss', 'total_loss'))
		pbar = tqdm(enumerate(val_loader), total=len(val_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

		for batch_idx, (images, targets) in pbar:
			
			images = images.to(rank)
			# print(images.shape)
			targets = [t.to(rank, non_blocking=True) for t in targets]

			# Forward pass
			outputs = model(images.permute(1,0,2,3,4))
			outputs = {'pred_logits':outputs[0], 'pred_boxes': outputs[1]}
			target = [{'labels': t[:,0], 'boxes':t[:,1:]} for t in targets]

			loss, batch_loss = compute_loss(outputs, target, criterion, nc)

			results = postprocessors['bbox'](outputs)
			res = {str(batch_idx): output for output in results}
			pbar.set_description(('%22.4g' * 5) % (batch_loss['cardinality_error'].item(), 
													batch_loss['loss_bbox'].item(), 
													batch_loss['loss_ce'].item(), 
													batch_loss['loss_giou'].item(), 
													loss.item()))
			tgt = []
			for i,t in enumerate(targets):
				tg = torch.zeros((len(t), 6))
				tg[:,1:] = t 
				tg[:,0] = i
				tgt.append(tg)

			tgt = torch.cat(tgt)

			opt = []
			for i in range(len(targets)):
				c = outputs['pred_logits'][i]
				b = outputs['pred_boxes'][i]

				op = torch.zeros((len(b), 7))

				cc = torch.argmax(c)

				c = c[cc<2]
				b = b[cc<2]

				if len(c)<1:
					opt.append(op)
					continue

				print(c)
				print(b)
				op[:,0] = i 
				op[:,1] = cc
				op[:,2:6] = b
				op[:,6] = torch.max(c)
				print(op)
				print('\n')
				opt.append(op)

			opt = torch.cat(opt)


			plot_images(images[:,1].cpu(), tgt.cpu(), fname=str(batch_idx)+'.png')
			# plot_images(images[:,1].cpu(), opt.cpu(), fname=str(batch_idx)+'pred.png')

			break
	return



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



def validate(rank, world_size, opt):
	setup(rank, world_size)
	# batch_size = 16
	# nc=2
	# r=3
	# space=1

	batch_size = opt.batch 
	nc = opt.nc
	r = opt.r 
	space = opt.space

	dataroot = opt.dataroot
	root = opt.root 

	weights = opt.weights

	val_dataset = get_dataset(rank, world_size, dataroot, 'val2', batch_size, r, space)
	
	# define detection model
	model = Dent(hidden_dim=256, num_class=2).to(rank)
	model = DDP(model, device_ids=[rank], find_unused_parameters=True)

	val_loader = get_loader(val_dataset, batch_size)

	ckptfile = root + weights + '.pth'
	ckpts = torch.load(ckptfile, map_location='cpu')
	model.load_state_dict(ckpts['model_state_dict'])
	best_accuracy = ckpts['best_val_acc']

	criterion_val,_ = loss_functions(nc, phase='val')
	# rank = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	epoch_validate(rank, model, val_loader, criterion_val, nc)
	cleanup()




def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/media/jakep/eye/scr/dent/', help='project root path')
    parser.add_argument('--dataroot', type=str, default='/media/jakep/eye/scr/pickle/', help='path to pickled dataset')
    parser.add_argument('--world_size', type=int, default=1, help='World size')
    parser.add_argument('--weights', type=str, default='outputs/detection_21', help='path to trained weights')
    
    parser.add_argument('--nc', type=int, default=2, help='number of classes')
    parser.add_argument('--r', type=int, default=3, help='number of adjacent images to stack')
    parser.add_argument('--space', type=int, default=1, help='Number of steps/ stride for next adjacent image block')
    parser.add_argument('--batch', type=int, default=16, help='validation batch size')

    return parser.parse_args()



if __name__ == '__main__':

	opt = arg_parse()
	mp.spawn(validate, args=(opt.world_size, opt), nprocs=opt.world_size, join=True)