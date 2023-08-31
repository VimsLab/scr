import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models import resnet50, ResNet50_Weights


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def forward(self, src, src_mask=None, src_key_padding_mask=None):
		output = src
		self_attn_weights = None
		if self.self_attn is not None:
			output,_ = self.self_attn(output, output, output, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
		
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
		
		return output #, self_attn_weights



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
			output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

		if convert_to_nested:
			output = output.to_padded_tensor(0.)

		if self.norm is not None:
			output = self.norm(output)
		
		return output


class Encoder(nn.Module):
	def __init__(self, hidden_dim=256, nheads=8, num_encoder_layers=6, dropout_rate=0.1):
		super().__init__()
		backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
		backbone.fc = torch.nn.Identity()
		backbone = nn.Sequential(*list(backbone.children())[:-2])

		for i,n in backbone.named_parameters():
			n.requires_grad=True
			
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
		print('Inputs',inputs.shape)
		# Get the features from the backbone
		features = self.backbone(inputs)
		print('after backbone',features.shape)

		if features.dim()==2:
			features = (features.unsqueeze(-1)).unsqueeze(-1)
			# print(features.shape)
		
		# Flatten the features and apply the linear embedding
		batch_size, channels, height, width = features.shape
		print('batch channel height width', features.shape)
		features = features.flatten(2).transpose(1, 2) # shape: (batch_size, num_patches, channels)
		encoder_embedding = self.linear_emb(features) # shape: (batch_size, num_patches, hidden_dim)
		# print(A)

		print('Encoder embedding in encoder', encoder_embedding.shape)
		# Add the positional encoding to the embeddings
		position_encoding = self.position_embedding.repeat(batch_size, 1, height, width).flatten(2).transpose(1, 2)
		print('position encoding', position_encoding.shape)
		encoder_embedding += position_encoding # shape: (batch_size, num_patches, hidden_dim)

		
		# Apply the transformer encoder
		encoder_outputs = self.encoder(encoder_embedding) # shape: (batch_size, seq_len, hidden_dim) #16 144 256
		
		return encoder_outputs, position_encoding

class PatchEmbedding(nn.Module):
	def __init__(self, image_size, patch_size, in_channels, embed_dim):
		super(PatchEmbedding, self).__init__()
		self.image_size = image_size
		self.patch_size = patch_size
		self.num_patches = (image_size // patch_size) ** 2
		self.patch_embed = nn.Conv2d(
			in_channels,
			embed_dim,
			kernel_size=patch_size,
			stride=patch_size
		)

	def forward(self, x):
		print('patch embedding input shape', x.shape)
		x = self.patch_embed(x)
		print('after patch embedding', x.shape)
		x = rearrange(x, 'b e (h) (w) -> b (h w) e')
		print('after rearranging', x.shape)
		return x


class MLP(nn.Module):
	def __init__(self, in_features, hidden_features, out_features):
		super(MLP, self).__init__()
		self.fc1 = nn.Linear(in_features, hidden_features)
		self.fc2 = nn.Linear(hidden_features, out_features)
		self.act = nn.GELU()

	def forward(self, x):
		x = self.fc1(x)
		x = self.act(x)
		x = self.fc2(x)
		return x


class Block(nn.Module):
	def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
		super(Block, self).__init__()
		self.norm1 = nn.LayerNorm(embed_dim)
		self.attn = nn.MultiheadAttention(embed_dim, num_heads)
		self.norm2 = nn.LayerNorm(embed_dim)
		self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim)

	def forward(self, x):
		residual = x
		x = self.norm1(x)
		x = rearrange(x, 'b n e -> n b e')
		x, _ = self.attn(x, x, x)
		x = rearrange(x, 'n b e -> b n e')
		x = x + residual

		residual = x
		x = self.norm2(x)
		x = self.mlp(x)
		x = x + residual
		return x

	
class Dent_Class(nn.Module):
	def __init__(self, in_channels, embed_dim, num_classes, num_layers, num_heads, mlp_ratio=4.0):
		super(Dent_Class, self).__init__()
		self.encoder = Encoder(hidden_dim=embed_dim, nheads=num_heads, num_encoder_layers=num_layers)
		self.combination = nn.MaxPool3d(kernel_size=(in_channels, 1, 1))
		self.blocks = nn.ModuleList([
			Block(embed_dim, num_heads, mlp_ratio) for _ in range(num_layers)
		])
		self.position_embedding = nn.Parameter(torch.randn(1, embed_dim, 1, 1))
		self.norm = nn.LayerNorm(embed_dim)
		self.fc = nn.Linear(embed_dim, num_classes)

	def forward(self, x):
		print('Initial', x.shape)
		e1, _ = self.encoder(x[0])
		e2, position = self.encoder(x[1])
		e3, _ = self.encoder(x[2])
		print('e1 and position shape', e1.shape, position.shape)
		e = torch.stack([e1, e2, e3])
		print('after combining e', e.shape)
		e = e.permute(1, 0, 2, 3)
		print('after rearranging', e.shape)
		e = self.combination(e).squeeze(1)
		print('after maxpooling', e.shape)
		e += position
		print(position.shape, e.shape)

		for block in self.blocks:
			e = block(e)
		
		print('after running the layers', e.shape)

		x = self.norm(e)
		print('after norm', x.shape)
		x = x.mean(dim=1)
		print('after mean', x.shape)
		x = self.fc(x)
		print('after fc', x.shape)
		return x


if __name__ == '__main__':
	# for pretraining
	a = torch.rand((32,3,256,576))
	b, c, h, w = a.shape
	encoder = Encoder(hidden_dim=768, nheads=8, num_encoder_layers=8)
	result, p=encoder(a)
	print('Results', result.shape, 'Position encoding', p.shape)

	print('\n\n')
	# for classification
	a = torch.rand((3,32,3,256,576))
	DC = Dent_Class(in_channels=c, embed_dim=768, num_classes=2, num_heads=8, num_layers=8)
	result = DC(a)
	print('Classification result', result.shape)

