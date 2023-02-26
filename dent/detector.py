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
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

from pretrain import *
from utils.dataloader import create_dataloader

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

def setup(rank, world_size):
    # Initialize the process group
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12526",
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


def calculate_metrics(outputs, targets):
	# Get number of classes
	num_classes = outputs.shape[1]
	
	# Calculate precision, recall, and f1 score for each class
	precision = torch.zeros(num_classes)
	recall = torch.zeros(num_classes)
	f1 = torch.zeros(num_classes)
	ap_05 = torch.zeros(num_classes)
	ap_095 = torch.zeros(num_classes)
	
	# Flatten the outputs and targets
	# outputs = outputs.flatten()
	# targets = targets.flatten()
	
	# Calculate precision, recall, and f1 score for each class
	for i in range(num_classes):
		# Extract predictions and targets for the current class
		class_outputs = outputs[targets == i]
		class_targets = targets[targets == i]
		
		# Calculate precision and recall
		tp = torch.sum(class_outputs == i)
		fp = torch.sum(class_outputs != i)
		fn = torch.sum(class_targets != i)
		precision[i] = tp / (tp + fp)
		recall[i] = tp / (tp + fn)
		
		# Calculate f1 score
		f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
		
		# Calculate average precision at 0.5 and 0.95
		precision_values, recall_values, _ = precision_recall_curve(class_targets.cpu(), class_outputs.cpu())
		ap_05[i] = average_precision_score(class_targets.cpu(), class_outputs.cpu(), average='binary', pos_label=i)
		ap_095[i] = 0
		for j in range(len(recall_values)):
			if recall_values[j] >= 0.95:
				ap_095[i] = max(ap_095[i], precision_values[j])
	
	return precision, recall, f1, ap_05, ap_095


class ObjDetect(nn.Module):
    def __init__(self, encoder_embedding, hidden_dim=256, num_queries=100, num_decoder_layers=6):
        super(DETRDecoder, self).__init__()

        self.encoder_embedding = encoder_embedding        
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_decoder_layers = num_decoder_layers
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(hidden_dim, nhead=8)
            for i in range(num_decoder_layers)
        ])
        
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, nhead=8),
            num_layers=num_decoder_layers
        )
        
        self.class_embed = nn.Linear(hidden_dim, num_queries)
        self.bbox_embed = nn.Linear(hidden_dim, 4 * num_queries)
        
    def forward(self, inputs, dec_seq_len):
        # Pad the memory tensor to match the sequence length of the decoder
        memory = self.encoder_embedding(inputs)
        printer([inputs, memory], ['Input image', 'Encoder output'])
        memory = nn.functional.pad(memory, (0, 0, 0, dec_seq_len - memory.shape[0]))
        
        # Generate positional encodings for the memory tensor
        pos_enc = nn.functional.embedding(
            torch.arange(dec_seq_len, device=memory.device).unsqueeze(1),
            torch.arange(self.hidden_dim, device=memory.device).unsqueeze(0)
        )
        printer([memory, pos_enc], ['Padded Encoder output', 'Position encoding'])
        pos_enc = pos_enc.repeat(1, memory.shape[1], 1)
        
        # Pass the memory tensor through the decoder layers
        tgt = torch.zeros((self.num_queries, memory.shape[1], self.hidden_dim), device=memory.device)
        for i in range(self.num_decoder_layers):
            tgt = self.transformer_layers[i](tgt, memory, pos_enc)
        output = self.transformer_decoder(tgt, memory, pos_enc)
        
        # Generate class and bbox embeddings from the output tensor
        class_output = self.class_embed(output)
        bbox_output = self.bbox_embed(output)
        
        # Reshape the bbox embeddings into the correct format
        bbox_outputs = bbox_output.view(-1, self.num_queries, 4)


        printer([pos_enc , tgt, output, class_output, bbox_output, bbox_outputs], 
			['Position Encoding after repeat', 'Target', 'Decoder output', 'Class prediction', 'Bbox prediction', 'Final bb0x prediction'])
        
        return class_output, bbox_outputs

# class MLP(nn.Module):
# 	def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
# 		super().__init__()
# 		layers = []
# 		layers.append(nn.Linear(input_dim, hidden_dim))
# 		layers.append(nn.ReLU())
# 		for i in range(num_layers-2):
# 			layers.append(nn.Linear(hidden_dim, hidden_dim))
# 			layers.append(nn.ReLU())
# 		layers.append(nn.Linear(hidden_dim, output_dim))
# 		self.mlp = nn.Sequential(*layers)
		
# 	def forward(self, inputs):
# 		return self.mlp(inputs)

# class ObjDetect(nn.Module):
# 	def __init__(self, embedding, num_classes, hidden_dim, num_decoder_layers=3, nheads=8, seq_length=64):
# 		super().__init__()
# 		self.embedding = embedding
# 		# self.position_enc = embedding.pos_encoder

# 		# Transformer decoder
# 		self.transformer_decoder = nn.TransformerDecoder(
# 			nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads),
# 			num_layers=num_decoder_layers,
# 		)

# 		# Class embedding and box position embedding
# 		self.query_embed = nn.Embedding(seq_length, hidden_dim)

# 		# Output layers
# 		self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
# 		self.bbox_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 4)


# 	def forward(self, x):
# 		# x: image
		
# 		# encoder output
# 		memory = self.embedding(x) # (src_seq_len, batch_size, hidden_dim)
# 		query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, x.shape[0], 1)  # shape: (tgt_seq_len, batch_size, hidden_dim)
# 		# print(query_embed.shape)
		
# 		# target sequence
# 		tgt = torch.zeros_like(query_embed)  # shape: (tgt_seq_len, batch_size, hidden_dim)
# 		# query_embed = self.query_embed.unsqueeze(1).repeat(1, bs, 1)  # query embeddings for each object query
# 		tgt[:, :, :query_embed.shape[-1]] = query_embed  # add query embeddings to tgt tensor

# 		tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[0]).to(x.device)
# 		# memory_mask = nn.Transformer.generate_square_subsequent_mask(memory.shape[0]).to(x.device)
# 		# print(tgt_mask.shape, memory_mask.shape)

# 		memory_mask = torch.triu(torch.full((tgt.shape[0], memory.shape[0]), float('-inf'), device=x.device), diagonal=1)
		
# 		hs = self.transformer_decoder(tgt, memory, tgt_mask, memory_mask)  # shape: (num_classes, batch_size, hidden_dim)
		
# 		# Classification and bbox regression output
# 		outputs_class = self.class_embed(hs)
# 		outputs_coord = self.bbox_embed(hs).sigmoid()

# 		printer([x, memory, query_embed, tgt, tgt_mask, memory_mask, hs], 
# 			['Input image', 'Encoder output', 'Query Embed', 'Target', 'Target mask', 'Memory mask', 'HS'])

# 		# Reshape outputs
# 		outputs_class = outputs_class.permute(1, 0, 2)
# 		outputs_coord = outputs_coord.permute(1, 0, 2).reshape(x.shape[0], tgt.shape[0], -1, 4)


# 		return outputs_class, outputs_coord


def compute_loss(outputs, targets, objthres=0.05):
	outputs_class, outputs_coord = outputs
	targets_class, targets_coord = targets[:,:,:1], targets[:,:,1:]

	# Calculate objectness score
	objectness = torch.softmax(outputs_class, dim=-1)[:, :, 0]
	outputs_class = outputs_class[objectness>objthres]
	outputs_coord = outputs_coord[objectness>objthres]

	# One-on-one mapping between predictions and targets using Hungarian algorithm
	# to find the lowest cost
	num_classes = outputs_class.shape[-1]
	cost_matrix = torch.cdist(outputs_coord, targets_coord, p=1)
	print(cost_matrix.shape)
	_, col_indices = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
	targets_class_mapped = targets_class[col_indices]
	targets_coord_mapped = targets_coord[col_indices]

	# Calculate classification and bounding box regression loss
	loss_class = F.cross_entropy(outputs_class.view(-1, num_classes), targets_class_mapped.flatten())
	loss_bbox = F.smooth_l1_loss(outputs_coord, targets_coord_mapped, reduction='sum') / outputs_coord.shape[0]

	# Total loss
	loss = loss_class + loss_bbox

	return loss, objectness


def detector_train_epoch(rank, model, optimizer, train_loader, epoch, epochs, running_loss=0):
	model.train()

	if rank==0:
		print(('\n' + '%22s' * 4) % ('Device', 'Epoch', 'GPU Mem', 'Loss'))

	pbar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:+10b}')

	for batch_idx, (img, targets) in pbar:
		img = img.to(rank, non_blocking=True)
		# targets = [{k: v.to(rank) for k, v in t.items()} for t in targets]
		targets = targets.to(rank, non_blocking=True)

		optimizer.zero_grad()

		# Forward pass
		outputs = model(img)
		loss, _ = compute_loss(outputs, targets)

		# Backward pass and optimization
		loss.backward()
		optimizer.step()

		# print statistics

		running_loss = running_loss+loss.item()

		if rank==0:
			mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
			pbar.set_description(('%22s'*3 + '%22.4g' * 1) % (f'{torch.cuda.current_device()}', f'{epoch}/{epochs - 1}', mem, running_loss/(batch_idx+1)))



def detector_validate(rank, world_size, model, val_loader, device, nc=2):
	model.eval()
	total_loss = 0.0
	total_correct = 0.0
	total_pred_boxes = []
	total_pred_classes = []
	total_target_boxes = []
	total_target_classes = []
	metrices = []
	
	with torch.no_grad():
		if rank==0:
			print(('\n' + '%22s' * 6) % ('Device', 'Correct', 'Accuracy', 'cls_loss', 'coord_loss', 'total_loss'))
		pbar = tqdm(enumerate(val_loader), total=len(val_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

		for i, (images, targets) in pbar:
			images = images.to(rank)
			targets = [{k: v.to(rank) for k, v in t.items()} for t in targets]
			
			# Forward pass
			outputs_class, outputs_coord = model(images)

			num_classes = outputs_class.shape[-1]
			
			# Calculate objectness score
			outputs_obj = outputs_class.softmax(dim=-1)[:, :, :-1]
			
			# Flatten outputs and targets
			outputs_obj_flat = outputs_obj.flatten(0, 1)
			outputs_coord_flat = outputs_coord.flatten(0, 1)
			target_boxes_flat = torch.cat([t['boxes'] for t in targets], dim=0)
			target_classes_flat = torch.cat([t['labels'] for t in targets], dim=0)
			num_targets = target_classes_flat.shape[0]

			# Get metrics
			val_metrics = calculate_metrics(outputs_obj_flat, target_classes_flat)
			metrices.append(torch.stack(val_metrics))
			
			# Apply one-on-one mapping between predictions and targets
			cost_matrix = torch.cdist(outputs_coord_flat, target_boxes_flat, p=1)
			cost_matrix[cost_matrix > 100] = 100 # Set a high cost for invalid matches
			row_indices, col_indices = linear_sum_assignment(cost_matrix.cpu().numpy())
			pred_boxes = outputs_coord_flat[row_indices]
			pred_classes = outputs_obj_flat[row_indices, col_indices]
			
			# Calculate validation loss
			target_obj = F.one_hot(target_classes_flat, num_classes=num_classes.float())
			target_obj[target_obj.sum(dim=0) == 0] = 1.0 / num_classes # Handle empty classes
			target_obj = target_obj[:, :-1]
			class_loss = F.cross_entropy(outputs_obj_flat, target_obj.flatten())
			coord_loss = F.smooth_l1_loss(outputs_coord_flat, target_boxes_flat)
			loss = class_loss + coord_loss
			total_loss += loss.item()
			
			# Calculate validation accuracy
			pred_classes = pred_classes.argmax(dim=-1)
			total_correct += (pred_classes == target_classes_flat[row_indices]).sum().item()
			
			# Save predictions and targets for visualization
			total_pred_boxes.append(pred_boxes.cpu())
			total_pred_classes.append(pred_classes.cpu())
			total_target_boxes.append(target_boxes_flat.cpu())
			total_target_classes.append(target_classes_flat.cpu())

			if rank==0:
				pbar.set_description(('%22s'*2 +'%22.4g' * 4) % (f'{torch.cuda.current_device()}', f'{total_correct}/{num_targets}', 
					total_correct/num_targets, class_loss/(batch_idx+1), coord_loss/(batch_idx+1), total_loss/(batch_idx+1)))

			
	avg_loss = total_loss / len(val_loader)
	avg_accuracy = total_correct / num_targets
	metrices = torch.stack(metrices)
	if rank==0:
		print('Metrices: ', metrices.shape, torch.mean(metrices, dim=0), torch.mean(metrices, dim=0).shape)
	return avg_loss, avg_accuracy, total_pred_boxes, total_pred_classes, total_target_boxes, total_target_classes



def detector(rank, world_size, root, dataroot):
	setup(rank, world_size)
	# define the siamese network model
	encoder = Encoder(hidden_dim=256, num_encoder_layers=6, nheads=8)
	siamese_net = DataParallel(SiameseNetwork(encoder))

	# load pre-trained weights
	ckpt = torch.load('best_pretrainer.pth', map_location='cpu')
	siamese_net.load_state_dict(ckpt['model_state_dict'])

	siamese_net = siamese_net.module

	# siamese_net = siamese_net.module
	siamese_net.eval()

	embedding = siamese_net.encoder

	# trainig params
	nc = 2
	hidden_dim = embedding.fc.out_features
	epochs = 2
	r = 3
	space = 1
	batch_size = 1

	# define detection model
	model = ObjDetect(embedding, hidden_dim=hidden_dim).to(rank)
	model = DDP(model, device_ids=[rank], find_unused_parameters=True)

	# declare optimizer and scheduler
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

	# get datasets
	train_loader, _ = get_dataset(rank, world_size, dataroot, 'train', batch_size, r, space)
	val_loader, _ = get_dataset(rank, world_size, dataroot, 'val', batch_size, r, space)


	best_fitness = 0
	
	for epoch in range(epochs):
		detector_train_epoch(rank, model, optimizer, train_loader, epoch, epochs)
		
		# Update the learning rate
		lr_scheduler.step()
		
		if rank==0:
			metrics = detector_validate(rank, model, val_loader)
			vloss, fitness = metrics[0], metrics[1]

			if fitness>=best_fitness:
				best_fitness = fitness
				save_path = root + 'detection_best.pth'
			else:
				save_path = root + 'detection_last.pth'

			# if rank==0:
			checkpoint = {
					'epoch': epoch,
					'model_state_dict': siamese_net.module.state_dict(),
					'optimizer_state_dict': optimizer.module.state_dict(),
					'best_val_acc': best_fitness,
				}
			torch.save(checkpoint, save_path)
	cleanup()
		
def get_dataset(rank, world_size,dataroot, task, batch_size, r, space, phases=["minitrain2", "minival2"]):
	phase = phases[task in phases]
	data_loader, dataset = create_dataloader(dataroot+phase,
											  576,
											  batch_size, 
											  rank=rank,                                   
											  cache='ram', # if opt.cache == 'val' else opt.cache,
											  workers=8,
											  phase='train',
											  shuffle=True,
											  r=r,
											  space=space)

	return data_loader, dataset



if __name__ == '__main__':

	root = '/media/jakep/eye/scr/dent/'
	dataroot = '/media/jakep/eye/scr/pickle/'

	# ddp
	world_size = 1
	mp.spawn(detector, args=(world_size, root, dataroot), nprocs=world_size, join=True)


	