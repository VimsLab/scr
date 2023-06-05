import os
import torch
import pickle
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from glob import glob
from tqdm import tqdm
from pathlib import Path
import torch.distributed as dist
from collections import OrderedDict
from matplotlib import pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP

from pretrain import *
from model.transformer import Encoder
from utils.util import (get_pickles, ids, fold_operation, split, load_one_pickle, plot_attention)





TQDM_BAR_FORMAT = '{desc} {n_fmt}/{total_fmt} [{elapsed} | {remaining} | {rate_fmt}]'


def printer(vals, names):
	print('\n')
	for val, name in zip(vals, names):
		print(f'{name}: {val.shape}')


def setup(rank, world_size):
    # Initialize the process group
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:20143",
        rank=rank,
        world_size=world_size
    )
    # Set the GPU to use
    torch.cuda.set_device(rank)


def run_inference(siamese_net, val_loader, device, gradcam, thres=0.5):
	"""
	"""
	siamese_net.eval()
	with torch.no_grad():
		total_loss = 0 
		corrects = 0
		tps = 0
		tns = 0
		total = 0

		print(('\n' + '%44s' + '%22s' * 4) % ('Correct', '(TP,P)', '(TN,N)', 'Accuracy', 'Loss'))
		pbar = tqdm(enumerate(val_loader), total=len(val_loader), bar_format=TQDM_BAR_FORMAT)

		for batch_idx, (x1, x2, targets, f1, f2) in pbar:
			x1 = x1.to(device, non_blocking=True)
			x2 = x2.to(device, non_blocking=True)
			targets = targets.to(device, non_blocking=True)


			

			# Forward pass
			embeddings1, embeddings2 = siamese_net(x1, x2)
			
			# print(atn1.shape, atn2.shape)

			loss = contrastive_focal_loss(device, embeddings1[:,-1], embeddings2[:,-1], targets)

			dist= get_distance(embeddings1[:,-1], embeddings2[:,-1])
			
			threshold = torch.ones_like(dist)*thres
			op = torch.relu(torch.sign(threshold-dist))

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

			# accumulate loss
			total_loss += loss.item()
			# ipt = [*x1, *x2]
			
			

			# plot_attention(x1, atn1, f1)



			pbar.set_description(('%44s' + '%22s'*2 +'%22.4g' * 2) % (correct, f'({tp},{p})', f'({tn},{n})', correct/(p+n), loss.item()))

			# if batch_idx > 4:
			# 	break


		# calculate average loss and accuracy
		avg_loss = total_loss / len(val_loader)
		accuracy = corrects / total


		print(('\n'+ '%44s') % ('Validation stats:'))
		print(('%44s' + '%22s' * 5) % ('Total', 'TP', 'TN', 'Incorrect', 'avg_acc', 'avg_loss'))
		print(('%44s' + '%22s' * 3 + "%22.4g"*2) % (total, f'{tps}/{corrects}', f'{tns}/{corrects}', total-corrects, accuracy, avg_loss))

	return avg_loss, accuracy



def infer(device, opt):

	batch_size = opt.batch_size 
	root = opt.root 
	folds = opt.folds
	fold = opt.f
	weights_path = 'runs/' + str(fold) + opt.weights_path
	phase = opt.val_folder

	setup(0, 1)

	# define the siamese network model
	encoder = Encoder(hidden_dim=256, num_encoder_layers=6, nheads=8)
	siamese_net = SiameseNetwork(encoder).to(device)
	

	# print(A)	

	siamese_net = DDP(siamese_net, device_ids=[0], find_unused_parameters=False)

	ckptfile = root + weights_path + '.pth'
	ckpt = torch.load(ckptfile, map_location='cpu')

	print(('\n'+ '%44s' + '%22s' * 5) % ('Training stats for model:', ckptfile, 
										f'Epochs:', ckpt['epoch'], 
										f'Best accuracy:', ckpt['best_val_acc']))

	ckpt = ckpt['model_state_dict']

	siamese_net.load_state_dict(ckpt)
	siamese_net.to(device)

	for name, layer in siamese_net.named_modules():
		if isinstance(layer, torch.nn.Conv2d):
			print(name, layer)	


	
	_, val = split(folds, fold)
	val_loader, _ = get_dataset(0, 0, val, phase=phase,transform=None, batch_size=batch_size, task='infer')


	run_inference(siamese_net, val_loader, device, gradcam)
	cleanup()


def arg_parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--root', type=str, default='/media/jakep/eye/scr/dent/', help='project root path')
	parser.add_argument('--weights_path', type=str, default='best_pretrainer', help='path to trained weights')
	parser.add_argument('--folds', type=int, default=5, help='number of dataset folds for training')
	parser.add_argument('--val_folder', type=str, default='val2', help='name of the directory containing validation samples') 
	parser.add_argument('--f', type=int, default=0, help='fold number to validate')
	parser.add_argument('--batch_size', type=int, default=2, help='batch size')

	return parser.parse_args()



if __name__ == '__main__':

	opt = arg_parse()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	infer(device, opt)