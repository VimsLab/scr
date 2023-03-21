import os
import re
import gc
import pdb
import torch
import pickle
import random
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
from validate import detector_validate

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



def printer(vals, names):
	print('\n')
	for val, name in zip(vals, names):
		print(f'{name}: {val.shape}')




def calculate_metrics(outputs, scores, targets, nc, eps=1e-3):
	# outputs: M [2]
	# scores: M [0.6]
	# targets: M [1]

	num_classes = nc
	
	# Calculate precision, recall, and f1 score for each class
	precision = torch.zeros(num_classes)
	recall = torch.zeros(num_classes)
	f1 = torch.zeros(num_classes)
	ap_05 = torch.zeros(num_classes)
	ap_095 = torch.zeros(num_classes)
	cf = torch.zeros((num_classes, 4))

	classes = torch.unique(targets)	
	# Calculate precision, recall, and f1 score for each class

	for i in classes:
		i = i.long()
		if len(outputs)<1:
			break
	
		# Extract predictions and targets for the current class
		tp = torch.sum(torch.logical_and(targets == i, outputs == i)).item()
		fp = torch.sum(torch.logical_and(targets != i, outputs == i)).item()
		tn = torch.sum(torch.logical_and(targets != i, outputs != i)).item()
		fn = torch.sum(torch.logical_and(targets == i, outputs != i)).item()

		# print(i, outputs, scores, targets, torch.logical_and(targets == i, outputs == i), tp, fp, tn, fn)
		
		cf[i-1] = torch.tensor([tp,tn,fp,fn])
		precision[i-1] = tp / (tp + fp+eps)
		recall[i-1] = tp / (tp + fn+eps)

		# Calculate f1 score
		f1[i-1] = 2 * precision[i-1] * recall[i-1] / (precision[i-1] + recall[i-1])

		if not torch.any(targets==i):
			continue

		class_targets = targets[targets==i]
		class_outputs = scores[targets==i]

		# Calculate average precision at 0.5 and 0.95
		precision_values, recall_values, _ = precision_recall_curve(class_targets.cpu(), class_outputs.float().cpu(), pos_label=int(i))
		ap_05[i-1] = average_precision_score(class_targets.cpu(), class_outputs.float().cpu(), average='samples', pos_label=int(i))

	return precision, recall, f1, ap_05, ap_095, cf



class MLP(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
		super().__init__()
		self.num_layers = num_layers
		h = [hidden_dim] * (num_layers -1)
		self.layers = nn.ModuleList(nn.Linear(n,k) for n,k in zip([input_dim] +h, h + [output_dim]))
		
	def forward(self, x):
		for i, layer in enumerate(self.layers):
			x = F.relu(layer(x)) if i<self.num_layers -1 else layer(x)
		return x



class ObjDetect(nn.Module):
	def __init__(self, encoder_embedding, hidden_dim=256, num_queries=48, num_decoder_layers=6, num_class=2):
		super().__init__()

		self.encoder_embedding = encoder_embedding        
		self.num_queries = num_queries
		self.hidden_dim = hidden_dim
		self.num_decoder_layers = num_decoder_layers
		self.num_class = num_class
		
		self.transformer_layers = nn.ModuleList([
			nn.TransformerDecoderLayer(hidden_dim, nhead=8)
			for i in range(num_decoder_layers)
		])
		
		self.transformer_decoder = nn.TransformerDecoder(
			nn.TransformerDecoderLayer(hidden_dim, nhead=8),
			num_layers=num_decoder_layers
		)

		self.class_embed = nn.Linear(hidden_dim, num_class+1)
		self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
		

		

	def forward(self, inputs, dec_seq_len=48):

		memory = self.encoder_embedding(inputs)
		tgt = torch.zeros((self.num_queries, memory.shape[1], self.hidden_dim), device=memory.device)

		# Generate masks
		tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[0]).to(memory.device)
		memory_mask = torch.triu(torch.full((tgt.shape[0], memory.shape[0]), float('-inf'), device=memory.device), diagonal=1)
		
		# Apply decoder layers
		for i in range(self.num_decoder_layers):
			tgt = self.transformer_layers[i](tgt, memory, tgt_mask, memory_mask)
		output = self.transformer_decoder(tgt, memory, tgt_mask, memory_mask)
		
		# Generate class and bbox embeddings from the output tensor
		class_output = self.class_embed(output)
		bbox_output = self.bbox_embed(output).sigmoid()
		
		# Reshape the bbox embeddings into the correct format
		class_output = class_output.view(-1, self.num_queries, self.num_class+1)
		bbox_output = bbox_output.view(-1, self.num_queries, 4)

		# printer([tgt, tgt_mask, memory_mask, output, class_output, bbox_output], 
		# 	['Target', 'Target Mask', 'Memory mask', 'Decoder output', 'Class prediction', 'Bbox prediction'])
		class_output = torch.softmax(class_output, dim=-1)
		return class_output, bbox_output


def get_focal_loss(pred, target, gamma=1):
	# pred contains probaility score for each class
	alpha = torch.tensor([1, 2.4, 1.8]).to(pred.device)

	ce_loss = nn.CrossEntropyLoss(reduction='none', weight=alpha)(pred.float(), target.long())
	pt = torch.exp(-ce_loss)
	focal_loss = (1 - pt) ** gamma * ce_loss
	return torch.mean(focal_loss)


def get_iou_loss(pred, target):
	iou_loss = complete_box_iou_loss(pred.float(), target.float(), reduction='mean')
	return iou_loss


def compute_loss(outputs, targets, nc=2):
	criterion,_ = loss_functions(nc, phase='train')
	criterion.train()
	
	loss_dict = criterion(outputs, targets) #.to(outputs.device)

	weight_dict = criterion.weight_dict
	losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

	return losses



def compute_loss_old(outputs, targets, objthres=0.45, phase='val'):
	outputs_class, outputs_coord = outputs
	targets_class = targets[:,0]
	targets_coord = targets[:,1:]

	nc = outputs_class.shape[-1]

	# Calculate objectness score
	objectness = torch.softmax(outputs_class.flatten(0,1), dim=-1)	
	indices = torch.argmax(objectness, dim=-1)						# predicted class index [N]
	topobjectness  = torch.max(objectness, dim=-1)[0]				# highest probability score of the predicted class index [N]

	outputs_class = objectness.view(-1, nc)							# probabilities [Nx3]
	outputs_coord = outputs_coord.view(-1, 4)

	idx_array = torch.arange(len(objectness))


	top_real_objects = topobjectness[topobjectness>objthres]		# probability score of predicted class after filtering low scores [M] [0.6]
	# real_objects = objectness[topobjectness>objthres]				
	indices = indices[topobjectness>objthres]						# class indices after filtering [M] [2]

	outputs_class = outputs_class[topobjectness>objthres]			# probability scores of all classes after the filtering [MX3] [0.2,0.2, 0.6]
	outputs_coord = outputs_coord[topobjectness>objthres]

	idx_array = idx_array[topobjectness>objthres]

	# One-on-one mapping between predictions and targets using Hungarian algorithm
	# to find the lowest cost
	cost_matrix = torch.cdist(outputs_coord.unsqueeze(0), targets_coord.unsqueeze(0), p=1).squeeze(0)
	row_indices, col_indices = linear_sum_assignment(cost_matrix.cpu().detach().numpy())

	op_class_mapped = outputs_class[row_indices]
	op_coord_mapped = outputs_coord[row_indices]
	top_real_objects = top_real_objects[row_indices]

	if len(row_indices)<len(targets_class):
		n = len(targets_class) - len(row_indices)
		extra_class = torch.zeros((n,nc)).to(targets_class.device)
		extra_class[:,0] = 1
		no_obj = torch.zeros((n)).to(targets_class.device)
		extra_box = torch.zeros((n, 4)).to(targets_class.device)
		op_class_mapped = torch.cat([op_class_mapped, extra_class])
		op_coord_mapped = torch.cat([op_coord_mapped, extra_box])
		top_real_objects= torch.cat([top_real_objects, no_obj])

	ce = nn.CrossEntropyLoss(reduction='mean')
	assert len(op_class_mapped)==len(targets_class), f'Lengths dont match'
	assert len(op_class_mapped)>0,f'empty'
	
	loss_class = get_focal_loss(op_class_mapped, targets_class)
	# loss_bbox = F.smooth_l1_loss(op_coord_mapped.unsqueeze(0), targets_coord.unsqueeze(0), reduction='mean') 
	loss_bbox = get_iou_loss(op_coord_mapped.unsqueeze(0), targets_coord.unsqueeze(0))

	# Total loss
	if phase=='train':
		return loss_class + loss_bbox

	return (loss_class, loss_bbox), (top_real_objects, op_coord_mapped), (indices[row_indices], idx_array[row_indices])




def detector_train_epoch(rank, model, optimizer, train_loader, epoch, epochs, nc, running_loss=0):
	model.train()
	gc.collect()

	if rank==0:
		print(('\n' + '%22s' * 4) % ('Device', 'Epoch', 'GPU Mem', 'Loss'))

	pbar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:+10b}')

	for batch_idx, (img, targets) in pbar:
		img = img.to(rank, non_blocking=True)
		targets = [t.to(rank, non_blocking=True) for t in targets]
		optimizer.zero_grad()
		# Forward pass
		outputs = model(img)
		outputs = {'pred_logits':outputs[0], 'pred_boxes': outputs[1]}
		targets = [{'labels': t[:,0], 'boxes':t[:,1:]} for t in targets]

		# loss = compute_loss(outputs, targets[:,1:], phase='train', objthres=0.001)
		loss = compute_loss(outputs, targets)

		# Backward pass and optimization
		loss.backward()
		optimizer.step()

		running_loss = running_loss+loss.item()

		if rank==0:
			mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
			pbar.set_description(('%22s'*3 + '%22.4g' * 1) % (f'{torch.cuda.current_device()}', f'{epoch}/{epochs - 1}', mem, running_loss/(batch_idx+1)))
			# if batch_idx<10 and epoch==0:
			# 	plot_images(img, targets, paths=None, fname=f'images/train_batch_{batch_idx}.jpg')
		
	return model




def detector_validate_old(rank, model, val_loader, nc):
	model.eval()
	total_loss = 0.0
	total_correct = 0.0
	total_incorrect = 0.0
	total_class_loss = 0.0
	total_coord_loss = 0.0
	num_targets = 0
	metrices = []
	fours = [] #torch.zeros((nc,4))
	
	with torch.no_grad():
		if rank==0:
			print(('\n' + '%22s' * 6) % ('Device', 'Correct', 'Incorrect', 'cls_loss', 'coord_loss', 'total_loss'))
		pbar = tqdm(enumerate(val_loader), total=len(val_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

		for batch_idx, (images, targets) in pbar:
			images = images.to(rank)
			targets = targets.to(rank, non_blocking=True)

			# Forward pass
			outputs = model(images)

			if len(targets>0):
				lossthings = compute_loss(outputs, targets[:,1:], phase='val', objthres=0.1)
				class_loss, coord_loss = lossthings[0]
				outputs_obj, outputs_coord = lossthings[1]
				class_index, array_index = lossthings[2]

				inter = torch.zeros(outputs[0].shape[:-1]).flatten()
				inter[array_index.long()] = 1
				inter = inter.view(outputs[0].shape[0], outputs[0].shape[1])
				batchid = torch.where(inter)[0].to(targets.device)

				op_scores = torch.softmax(outputs[0], dim=-1)[inter==1]
				# op_class = torch.argmax(op_scores, dim=-1)
				op_class_score, op_class = torch.max(op_scores, dim=-1)
				op_coord = outputs[1][inter==1]


				# print(op_coord.shape, op_class.shape, batchid.shape, op_scores.shape)

				target_classes = targets[:,1]

				# Get metrics
				val_metrics = calculate_metrics(class_index, outputs_obj, target_classes, nc)
				cf = val_metrics[-1]
				fours.append(cf)

				metrices.append(torch.stack(val_metrics[:-1]))
				
				total_class_loss = total_class_loss + class_loss.item()
				total_coord_loss += coord_loss.item()
				total_loss += (class_loss + coord_loss).item()
				
				# print(cf)
				correct = torch.sum(cf[:,0] + cf[:,1]).item()
				incorrect = torch.sum(cf[:,2] + cf[:,3]).item()

				total_correct += correct 
				total_incorrect += incorrect
				
				if rank==0:
					pbar.set_description(('%22s'*3 +'%22.4g' * 3) % (f'{torch.cuda.current_device()}', f'{correct}/{correct + incorrect}', 
						f'{incorrect}/{correct + incorrect}', class_loss.item(), coord_loss.item(), (class_loss + coord_loss).item()))

				op = torch.cat([batchid.view(-1,1), op_class.view(-1,1), op_coord, op_class_score.view(-1,1)], dim=-1)

				if rank==0 and batch_idx<10:
					plot_images(images[:,1:2,:,:], targets, paths=None, fname=f'images/val_batch_{batch_idx}_labels.jpg')
					plot_images(images[:,1:2,:,:], op, paths=None, fname=f'images/val_batch_{batch_idx}_pred.jpg')


	avg_loss = total_loss / len(val_loader)
	avg_accuracy = total_correct / (total_correct+total_incorrect)
	metrices = torch.mean(torch.stack(metrices), dim=0)
	fours = torch.sum(torch.stack(fours), dim=0)
	names = ['Fovea', 'SCR']
	loss = [total_class_loss, total_coord_loss]
	fitness = get_fitness(torch.mean(metrices, dim=-1))
	acc = [fitness, avg_loss]

	if rank==0:
		print(('\n'+'%44s' + '%22s'*7) % ('[tp,tn,fp,fn]','Precision', 'Recall', 'F1', 'AP@0.5', 'AP@0.95', 'cls_loss/obj_loss', 'fitness/loss'))
		for v in range(len(names)):
			print(('%22s'*2 + '%22.4g'*7 + '\n') % (names[v], f'[{fours[v][0].item()},{fours[v][1]},{fours[v][2]},{fours[v][3]}]',
				metrices[0][v].item(), metrices[1][v].item(), metrices[2][v].item(), 
				metrices[3][v].item(), metrices[4][v].item(), loss[v], acc[v] ))
		
	return avg_loss, avg_accuracy, fitness #total_pred_boxes, total_pred_classes, total_target_boxes, total_target_classes




def get_fitness(metrics):
	p,r,_,ap,ap95 = metrics
	return p*0.2+r*0.2+ap*0.2+ap95*0.4

def detector(rank, world_size, root, dataroot, pretraining=False, pretrained_weights_path='best_pretrainer.pth'):
	setup(rank, world_size)
	# trainig params
	nc = 100
	epochs = 152
	r = 3
	space = 1
	batch_size = 64
	val_batch_size = 16

	# if rank>-1:
	train_data = get_dataset(rank, world_size, dataroot, 'train2', batch_size, r, space)
	val_dataset = get_dataset(rank, world_size, dataroot, 'val2', batch_size, r, space)
	gc.collect()
	
	# define the siamese network model
	encoder = Encoder(hidden_dim=256, num_encoder_layers=6, nheads=8)

	if pretraining:
		siamese_net = DataParallel(SiameseNetwork(encoder))
		# load pre-trained weights
		ckpt = torch.load(pretrained_weights_path, map_location='cpu')
		siamese_net.load_state_dict(ckpt['model_state_dict'])
		siamese_net = siamese_net.module
		siamese_net.eval()
		embedding = siamese_net.module.encoder
	
	embedding = encoder
	hidden_dim = embedding.hidden_dim

	for param in embedding.parameters():
		param.requires_grad=True

	# define detection model
	model = ObjDetect(embedding, hidden_dim=hidden_dim, num_class=2).to(rank)
	model = DDP(model, device_ids=[rank], find_unused_parameters=False)

	# declare optimizer and scheduler
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

	# if rank==0:
		# val_loader, _ = get_dataset(rank, world_size, dataroot, 'val', batch_size, r, space)

	train_loader = get_loader(train_data, batch_size)
	val_loader = get_loader(val_dataset, val_batch_size)

	# torch.distributed.barrier()
	best_fitness = 0
	# del ckpt 
	del train_data
	torch.cuda.empty_cache()
	gc.collect()
	
	for epoch in range(epochs):
		model = detector_train_epoch(rank, model, optimizer, train_loader, epoch, epochs, nc)
		
		# Update the learning rate
		lr_scheduler.step()

		detector_validate(rank, model, val_loader, nc)
		
		# metrics = detector_validate(rank, model, val_loader, nc)
		# vloss, acc, fitness = metrics

		# # torch.distributed.barrier()

		# if fitness>=best_fitness:
		# 	best_fitness = fitness
		# 	save_path = root + 'outputs/detection_best.pth'
		# else:
		# 	save_path = root + 'outputs/detection_last.pth'
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





if __name__ == '__main__':

	root = '/media/jakep/eye/scr/dent/'
	dataroot = '/media/jakep/eye/scr/pickle/'

	# ddp
	world_size = 2
	mp.spawn(detector, args=(world_size, root, dataroot), nprocs=world_size, join=True)


	