import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import List, Dict

class DETREncoder(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, num_encoder_layers: int = 6, dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    def forward(self, src: Tensor, mask: Tensor) -> Tensor:
        return self.encoder(src, mask)

class DETRDecoderWithMask(nn.Module):
    def __init__(self, decoder: nn.Module, num_classes: int, n_queries: int = 100):
        super().__init__()
        self.decoder = decoder
        self.class_embed = nn.Linear(decoder.d_model, num_classes + 1)
        self.bbox_embed = MLP(decoder.d_model, decoder.d_model, 4, 3)

    def forward(self, memory: Tensor, memory_mask: Tensor, tgt: Tensor, tgt_mask: Tensor) -> Dict[str, Tensor]:
        hs = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        return {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim if layer == 0 else hidden_dim, hidden_dim if layer < num_layers - 1 else output_dim)
            for layer in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x




class DETR(nn.Module):
    """DETR object detection model"""

    def __init__(self, num_classes: int, hidden_dim: int = 256, nheads: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, **kwargs):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = torch.hub.load('facebookresearch/detectron2', 'resnet50', pretrained=True)

        # create feature map backbone
        self.in_features = ['res4']  # the name of the feature map that we will use to perform the detection
        self.feature_extractor = torch.nn.Sequential(*list(self.backbone.children())[:-2])

        # create position embeddings
        self.position_embedding = nn.Parameter(torch.randn(1, hidden_dim, 100, 100))

        # create transformer encoder
        self.transformer_encoder = DETREncoder(hidden_dim, nheads, num_encoder_layers)

        # create transformer decoder
        self.query_pos = nn.Parameter(torch.randn(100, hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, nheads, dim_feedforward=hidden_dim)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        # create final output layers
        self.decoder_with_mask = DETRDecoderWithMask(self.transformer_decoder, num_classes)

        # create the output layer for masks
        self.mask_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )

    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # extract features from the backbone
        features = self.feature_extractor(inputs['image'])

        # apply positional embeddings
        pos_embed = self.position_embedding[:, :, :features.shape[-2], :features.shape[-1]]
        features = features + pos_embed

        # encode features with transformer encoder
        src = features.flatten(2).permute(2, 0, 1)
        mask = inputs['mask'] if 'mask' in inputs else None
        memory = self.transformer_encoder(src, mask=mask)

        # decode objects with transformer decoder
        query_embed = self.query_pos.unsqueeze(1).repeat(1, inputs['bbox'].shape[0], 1)
        tgt = torch.zeros_like(query_embed)
        tgt[:, :, :self.decoder_with_mask.decoder.d_model] = query_embed
        tgt = tgt + self.decoder_with_mask.bbox_embed[0](inputs['bbox'])
        tgt_mask = self.generate_decoder_mask(inputs['bbox'])

        outputs = self.decoder_with_mask(memory, mask, tgt, tgt_mask)
        outputs['pred_masks'] = self.mask_head(memory.permute(1, 2, 0).view(memory.shape[1], memory.shape[0], *features.shape[-2:])).squeeze(1).sigmoid()
        return outputs

    
    def generate_decoder_mask(self, bbox: Tensor) -> Tensor:
        """Generate mask to prevent attention between decoded objects."""
        mask = bbox.sum(dim=-1) == 0
        mask = torch.triu(mask, diagonal=1)
        return mask

