import os
import re
import gc
import pdb
import copy
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
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from torchvision.ops import complete_box_iou_loss
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

from utils.plots import plot_images
from utils.dataloader import create_dataloader
from utils.general import xywh2xyxy, xyxy2xywh


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

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def forward(self, src, src_mask=None, src_key_padding_mask=None):
		output = src
		self_attn_weights = None
		
		if self.self_attn is not None:
			output, self_attn_weights = self.self_attn(output, output, output, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
		
		if self.norm1 is not None:
			output = self.norm1(output)
		
		if self.dropout is not None:
			output = self.dropout(output)
		
		residual = output
		
		if self.linear1 is not None:
			output = self.linear1(output)
		
		if self.activation is not None:
			output = self.activation(output)
		
		if self.dropout is not None:
			output = self.dropout(output)
		
		if self.linear2 is not None:
			output = self.linear2(output)
		
		if self.norm2 is not None:
			output = self.norm2(output)
		
		output += residual
		
		return output, self_attn_weights



class CustomTransformerEncoder(nn.TransformerEncoder):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
		src_key_padding_mask = F._canonical_mask(
			mask=src_key_padding_mask,
			mask_name="src_key_padding_mask",
			other_type=F._none_or_dtype(mask),
			other_name="mask",
			target_type=src.dtype
		)

		output = src
		convert_to_nested = False
		first_layer = self.layers[0]
		src_key_padding_mask_for_layers = src_key_padding_mask
		why_not_sparsity_fast_path = ''
		str_first_layer = "self.layers[0]"
		if not isinstance(first_layer, torch.nn.TransformerEncoderLayer):
			why_not_sparsity_fast_path = f"{str_first_layer} was not TransformerEncoderLayer"
		elif first_layer.norm_first :
			why_not_sparsity_fast_path = f"{str_first_layer}.norm_first was True"
		elif first_layer.training:
			why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
		elif not first_layer.self_attn.batch_first:
			why_not_sparsity_fast_path = f" {str_first_layer}.self_attn.batch_first was not True"
		elif not first_layer.self_attn._qkv_same_embed_dim:
			why_not_sparsity_fast_path = f"{str_first_layer}.self_attn._qkv_same_embed_dim was not True"
		elif not first_layer.activation_relu_or_gelu:
			why_not_sparsity_fast_path = f" {str_first_layer}.activation_relu_or_gelu was not True"
		elif not (first_layer.norm1.eps == first_layer.norm2.eps) :
			why_not_sparsity_fast_path = f"{str_first_layer}.norm1.eps was not equal to {str_first_layer}.norm2.eps"
		elif not src.dim() == 3:
			why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
		elif not self.enable_nested_tensor:
			why_not_sparsity_fast_path = "enable_nested_tensor was not True"
		elif src_key_padding_mask is None:
			why_not_sparsity_fast_path = "src_key_padding_mask was None"
		elif (((not hasattr(self, "mask_check")) or self.mask_check)
				and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
			why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
		elif output.is_nested:
			why_not_sparsity_fast_path = "NestedTensor input is not supported"
		elif mask is not None:
			why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
		elif first_layer.self_attn.num_heads % 2 == 1:
			why_not_sparsity_fast_path = "num_head is odd"
		elif torch.is_autocast_enabled():
			why_not_sparsity_fast_path = "autocast is enabled"

		if not why_not_sparsity_fast_path:
			tensor_args = (
				src,
				first_layer.self_attn.in_proj_weight,
				first_layer.self_attn.in_proj_bias,
				first_layer.self_attn.out_proj.weight,
				first_layer.self_attn.out_proj.bias,
				first_layer.norm1.weight,
				first_layer.norm1.bias,
				first_layer.norm2.weight,
				first_layer.norm2.bias,
				first_layer.linear1.weight,
				first_layer.linear1.bias,
				first_layer.linear2.weight,
				first_layer.linear2.bias,
			)

			if torch.overrides.has_torch_function(tensor_args):
				why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
			elif not (src.is_cuda or 'cpu' in str(src.device)):
				why_not_sparsity_fast_path = "src is neither CUDA nor CPU"
			elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
				why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
											  "input/output projection weights or biases requires_grad")

			if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
				convert_to_nested = True
				output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
				src_key_padding_mask_for_layers = None

		# Prevent type refinement
		make_causal = (is_causal is True)

		if is_causal is None:
			if mask is not None:
				sz = mask.size(0)
				causal_comparison = torch.triu(
					torch.ones(sz, sz, device=mask.device) * float('-inf'), diagonal=1
				).to(mask.dtype)

				if torch.equal(mask, causal_comparison):
					make_causal = True

		is_causal = make_causal

		for mod in self.layers:
			output, atn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

		if convert_to_nested:
			output = output.to_padded_tensor(0.)

		if self.norm is not None:
			output = self.norm(output)
		
		return output


class Encoder(nn.Module):
	def __init__(self, hidden_dim=256, nheads=8, num_encoder_layers=6, dropout_rate=0.1):
		super().__init__()
		backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
		backbone = nn.Sequential(*list(backbone.children())[:-2])
		for i in backbone.parameters():
			i.requires_grad=False
		self.backbone = backbone
		self.hidden_dim = hidden_dim
		self.nheads = nheads
		self.num_encoder_layers = num_encoder_layers
		self.dropout_rate = dropout_rate
		
		# Create a positional encoding module
		self.position_embedding = nn.Parameter(torch.randn(1, hidden_dim, 1, 1))
		
		# Create a linear layer for embedding the encoder features
		self.linear_emb = nn.Linear(2048, hidden_dim)
		
		# Create a transformer encoder
		encoder_layer = CustomTransformerEncoderLayer(d_model=hidden_dim, nhead=nheads, dropout=dropout_rate, batch_first=True)
		self.encoder = CustomTransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
		

	def forward(self, inputs):
		# Get the features from the backbone
		features = self.backbone(inputs)
		
		# Flatten the features and apply the linear embedding
		batch_size, channels, height, width = features.shape
		features = features.flatten(2).transpose(1, 2) # shape: (batch_size, num_patches, channels)
		encoder_embedding = self.linear_emb(features) # shape: (batch_size, num_patches, hidden_dim)

		
		# Add the positional encoding to the embeddings
		position_encoding = self.position_embedding.repeat(batch_size, 1, height, width).flatten(2).transpose(1, 2)
		encoder_embedding += position_encoding # shape: (batch_size, num_patches, hidden_dim)

		
		# Apply the transformer encoder
		encoder_outputs = self.encoder(encoder_embedding) # shape: (batch_size, seq_len, hidden_dim) #16 144 256
		
		return encoder_outputs




class TransformerDecoder(nn.Module):
	__constants__ = ['norm']

	def __init__(self, decoder_layer, num_layers, norm=None):
		super(TransformerDecoder, self).__init__()
		self.layers = _get_clones(decoder_layer, num_layers)
		self.num_layers = num_layers
		self.norm = norm

	def forward(self, tgt, memory):
		
		output = tgt

		for mod in self.layers:
			output, atn = mod(output, memory)

		if self.norm is not None:
			output = self.norm(output)

		return output, atn

class TransformerDecoderLayer(nn.Module):
	__constants__ = ['batch_first', 'norm_first']

	def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
				 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
				 device=None, dtype=None) -> None:
		factory_kwargs = {'device': device, 'dtype': dtype}
		super(TransformerDecoderLayer, self).__init__()
		self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
											**factory_kwargs)
		self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
												 **factory_kwargs)
		# Implementation of Feedforward model
		self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
		self.dropout = Dropout(dropout)
		self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

		self.position_encoding = nn.Parameter(torch.randn(1, 1, 1))

		self.norm_first = norm_first
		self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
		self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
		self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
		self.dropout1 = Dropout(dropout)
		self.dropout2 = Dropout(dropout)
		self.dropout3 = Dropout(dropout)
		
		self.activation = F.relu

	def __setstate__(self, state):
		if 'activation' not in state:
			state['activation'] = F.relu
		super(TransformerDecoderLayer, self).__setstate__(state)


	def forward(self, tgt, memory):
		
		n,b,le,hdim = memory.shape

		m = torch.max(memory, dim=0)[0]
		v = m 
		k = memory[1] + self.position_encoding.repeat(b, 1, hdim)
		x = tgt
		x = self.norm1(x + self._sa_block(x))
		y = x.clone()
		x = self.norm2(x + self._mha_block(x, k, v))
		x = self.norm3(x + self._ff_block(x))

		return x, y


	# self-attention block
	def _sa_block(self, x):
		x = self.self_attn(x, x, x)[0]
		return self.dropout1(x)

	# multihead attention block
	def _mha_block(self, x,k,v):
		x = self.multihead_attn(x, k, v)[0]
		return self.dropout2(x)

	# feed forward block
	def _ff_block(self, x):
		x = self.linear2(self.dropout(self.activation(self.linear1(x))))
		return self.dropout3(x)

def _get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
	if activation == "relu":
		return F.relu
	elif activation == "gelu":
		return F.gelu

	raise RuntimeError("activation should be relu/gelu, not {}".format(activation))



class Decoder(nn.Module):
	def __init__(self, hidden_dim=256, num_queries=16, num_decoder_layers=6, num_class=2):
		super().__init__()

		self.num_queries = num_queries
		self.hidden_dim = hidden_dim
		self.num_decoder_layers = num_decoder_layers
		self.num_class = num_class

		self.decoder_layer = TransformerDecoderLayer(hidden_dim, nhead=8)
		self.decoder = TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

		self.class_embed = nn.Linear(hidden_dim, num_class+1)
		self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
		

		

	def forward(self, memory):

		n, b, le, hdim = memory.shape
		tgt = torch.zeros((b, self.num_queries, self.hidden_dim), device=memory.device)

		# dl = self.decoder_layer(tgt, memory)
		output, atn = self.decoder(tgt, memory)

		# Generate class and bbox embeddings from the output tensor
		class_output = self.class_embed(output)
		bbox_output = self.bbox_embed(output).sigmoid()
		
		# Reshape the bbox embeddings into the correct format
		class_output = class_output.view(-1, self.num_queries, self.num_class+1)
		bbox_output = bbox_output.view(-1, self.num_queries, 4)

		class_output = torch.softmax(class_output, dim=-1)
		return class_output, bbox_output




class Dent(nn.Module):
	def __init__(self, hidden_dim=256, nheads=8, 
				num_encoder_layers=6, dropout=0.1, 
				num_queries=16, num_decoder_layers=6, num_class=2):
		super().__init__()
		self.encoder1=Encoder(hidden_dim, nheads, num_encoder_layers, dropout)
		self.encoder2=Encoder(hidden_dim, nheads, num_encoder_layers, dropout)
		self.encoder3=Encoder(hidden_dim, nheads, num_encoder_layers, dropout)

		self.decoder=Decoder(hidden_dim, num_queries, num_decoder_layers, num_class)


	def forward(self, inputs):	# (3,b,3,h,w)
		e1=self.encoder1(inputs[0])
		e2=self.encoder2(inputs[1])
		e3=self.encoder3(inputs[2])

		e=torch.stack([e1,e2,e3])
		outputs=self.decoder(e)

		return outputs


class Dent_Pt(nn.Module):
	def __init__(self, encoder,hidden_dim=256, nheads=8, 
				num_encoder_layers=6, dropout=0.1, 
				num_queries=16, num_decoder_layers=6, num_class=2):
		super().__init__()
		self.encoder = encoder
		self.decoder=Decoder(hidden_dim, num_queries, num_decoder_layers, num_class)


	def forward(self, inputs):	# (3,b,3,h,w)
		e1=self.encoder(inputs[0])
		e2=self.encoder(inputs[1])
		e3=self.encoder(inputs[2])

		e = torch.stack([e1,e2,e3])
		(op1, op2) = self.decoder(e)

		return op1, op2



