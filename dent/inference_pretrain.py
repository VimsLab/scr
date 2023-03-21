import os
import re
import torch
import pickle
import random
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
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

from pretrain import *
from transformer import Encoder
from utils.dataloader import create_dataloader


def printer(vals, names):
	print('\n')
	for val, name in zip(vals, names):
		print(f'{name}: {val.shape}')



def run_inference(siamese_net, val_loader, device, thres=4):
	"""
	thresh (float): distance threshold. Distance between two embeddings should be less than this value for a match
	"""
	with torch.no_grad():
		total_loss = 0 
		corrects = 0
		tps = 0
		tns = 0
		total = 0

		print(('\n' + '%22s' * 5) % ('Correct', '(TP,P)', '(TN,N)', 'Accuracy', 'Loss'))
		pbar = tqdm(enumerate(val_loader), total=len(val_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

		for batch_idx, (x1, x2, f1, f2, targets) in pbar:
			x1 = x1.to(device, non_blocking=True)
			x2 = x2.to(device, non_blocking=True)
			targets = targets.to(device, non_blocking=True)

			# Forward pass
			embeddings1, embeddings2 = siamese_net(x1, x2)

			loss = contrastive_focal_loss(0, embeddings1[:,-1], embeddings2[:,-1], targets)

			# distance between the embeddings for each batch (score)
			dist = torch.norm((embeddings1[:,-1] - embeddings2[:,-1]), 2, dim=-1)
			# dist = dist.sigmoid()
			
			threshold = torch.ones_like(dist)*thres
			# if dist < threshold op = 1 else op = 0
			op = torch.relu(torch.sign(threshold-dist))

			print([(d.item(), o.item(),t.item()) for d,o,t in zip(dist, op, targets)])

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

			pbar.set_description(('%22s'*3 +'%22.4g' * 2) % (correct, f'({tp},{p})', f'({tn},{n})', correct/total, loss.item()))


		# calculate average loss and accuracy
		avg_loss = total_loss / len(val_loader)
		accuracy = corrects / total

		print(('\n'+ '%22s') % ('Validation stats:'))
		print(('%22s' * 6) % ('Total', 'TP', 'TN', 'Incorrect', 'avg_acc', 'avg_loss'))
		print(('%22s' * 4 + "%22.4g"*2) % (total, f'{tps}/{corrects}', f'{tns}/{corrects}', total-corrects, accuracy, avg_loss))

	return avg_loss, accuracy


def infer(root, dataroot, device, pretrained_weights_path='best_pretrainer.pth'):
	# define the siamese network model
	encoder = Encoder(hidden_dim=256, num_encoder_layers=6, nheads=8)
	siamese_net = DataParallel(SiameseNetwork(encoder))

	# load pre-trained weights
	ckpt = torch.load(pretrained_weights_path, map_location='cpu')
	# siamese_net.load_state_dict(ckpt['model_state_dict'])
	# # siamese_net = siamese_net.module

	# for param in siamese_net.parameters():
	# 	param.requires_grad=False

	# siamese_net.eval()
	siamese_net.to(device)

	batch_size = 8

	# get dataset
	val_loader, _ = get_dataset(0, 0, dataroot, 
								phase='val2', lim=4, 
								transform=tx()['val'], 
								batch_size=batch_size)
	run_inference(siamese_net, val_loader, device)



if __name__ == '__main__':

	root = '/media/jakep/eye/scr/dent/'
	dataroot = '/media/jakep/eye/scr/pickle/'

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	infer(root, dataroot, device)