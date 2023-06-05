import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import complete_box_iou_loss

from .matching_loss import build_matcher, box_cxcywh_to_xyxy, generalized_box_iou, box_iou_loss



def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = torch.distributed.get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        torch.distributed.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict




class SetCriterion(nn.Module):
	""" This class computes the loss for DETR.
	The process happens in two steps:
		1) we compute hungarian assignment between ground truth boxes and the outputs of the model
		2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
	"""
	def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
		""" Create the criterion.
		Parameters:
			num_classes: number of object categories, omitting the special no-object category
			matcher: module able to compute a matching between targets and proposals
			weight_dict: dict containing as key the names of the losses and as values their relative weight.
			eos_coef: relative classification weight applied to the no-object category
			losses: list of all the losses to be applied. See get_loss for list of available losses.
		"""
		super().__init__()
		self.num_classes = num_classes
		self.matcher = matcher
		self.weight_dict = weight_dict
		self.eos_coef = eos_coef
		self.losses = losses
		empty_weight = torch.ones(self.num_classes + 1).to(torch.distributed.get_rank())
		empty_weight[-1] = self.eos_coef
		self.register_buffer('empty_weight', empty_weight)

	# @torch.no_grad()
	# def loss_cardinality(self, outputs, targets, indices, num_boxes):
	# 	""" Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
	# 	This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
	# 	"""
	# 	pred_logits = outputs['pred_logits']
	# 	device = pred_logits.device
	# 	tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
	# 	# Count the number of predictions that are NOT "no-object" (which is the last class)
	# 	card_pred = (pred_logits.argmax(-1) != 0).sum(1)
		
	# 	card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
	# 	losses = {'cardinality_error': card_err}
	# 	return losses
	
	def loss_nihuglabels(self, outputs, targets, indices, num_boxes, alpha=None, gamma=3):
		assert 'pred_logits' in outputs
		# src_logits = outputs['pred_logits']

		idx = self._get_src_permutation_idx(indices)
		src_logits = outputs['pred_logits'][idx]#.unsqueeze(0)

		target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
		target_classes = target_classes_o.long().view(-1,1)#.unsqueeze(0).long()

		alpha = torch.tensor([self.eos_coef, 0.55, 0.4]).type_as(src_logits).to(src_logits.device)
		logpt = torch.log(src_logits).gather(1, target_classes)
		logpt = logpt.view(-1)
		pt = torch.exp(logpt)

		alpha = alpha.gather(0, target_classes.view(-1))
		loss_ce = -1 * alpha * (1-pt) ** gamma * logpt
		loss_ce = loss_ce.sum()/num_boxes


		# print(loss_ce, loss_ce.shape)
		# print((loss_ce>0).sum(), loss_ce.mean())
		losses = {'loss_ce': loss_ce}

		# if (not len(target_classes)) or (not len(src_logits)):
		# 	losses = {'loss_ce': 1e-3*torch.ones_like(loss_ce)}

		return losses


	def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
		"""Classification loss (NLL)
		targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
		"""
		assert 'pred_logits' in outputs
		# src_logits = outputs['pred_logits']

		idx = self._get_src_permutation_idx(indices)
		src_logits = outputs['pred_logits'][idx]#.unsqueeze(0)

		target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
		target_classes = target_classes_o.long()#.unsqueeze(0).long()

		weights = torch.tensor([self.eos_coef, 0.6, 0.35]).to(src_logits.device)

		
		loss_ce = F.nll_loss(torch.log(src_logits), target_classes, weight=weights, reduction='none')
		loss_ce = loss_ce.sum()/num_boxes
		losses = {'loss_ce': loss_ce}

		return losses
	



	def loss_boxes(self, outputs, targets, indices, num_boxes):
		"""Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
		   targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
		   The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
		"""
		assert 'pred_boxes' in outputs
		idx = self._get_src_permutation_idx(indices)
		src_boxes = outputs['pred_boxes'][idx]
		target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
		
		loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

		losses = {}
		losses['loss_bbox'] = loss_bbox.sum() / num_boxes

		loss_iou = complete_box_iou_loss(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes), reduction='none')
		losses['loss_giou'] = loss_iou.sum() / num_boxes

		# loss_giou = 1 - torch.diag(generalized_box_iou(
		# 	box_cxcywh_to_xyxy(src_boxes),
		# 	box_cxcywh_to_xyxy(target_boxes)))
		# losses['loss_giou'] = loss_giou.sum() / num_boxes

		return losses
	
	

	def loss_masks(self, outputs, targets, indices, num_boxes):
		"""Compute the losses related to the masks: the focal loss and the dice loss.
		   targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
		"""
		assert "pred_masks" in outputs

		src_idx = self._get_src_permutation_idx(indices)
		tgt_idx = self._get_tgt_permutation_idx(indices)
		src_masks = outputs["pred_masks"]
		src_masks = src_masks[src_idx]
		masks = [t["masks"] for t in targets]
		# TODO use valid to mask invalid areas due to padding in loss
		target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
		target_masks = target_masks.to(src_masks)
		target_masks = target_masks[tgt_idx]

		# upsample predictions to the target size
		src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
								mode="bilinear", align_corners=False)
		src_masks = src_masks[:, 0].flatten(1)

		target_masks = target_masks.flatten(1)
		target_masks = target_masks.view(src_masks.shape)
		losses = {
			"loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
			"loss_dice": dice_loss(src_masks, target_masks, num_boxes),
		}
		return losses

	def _get_src_permutation_idx(self, indices):
		# permute predictions following indices
		batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
		src_idx = torch.cat([src for (src, _) in indices])
		return batch_idx, src_idx

	def _get_tgt_permutation_idx(self, indices):
		# permute targets following indices
		batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
		tgt_idx = torch.cat([tgt for (_, tgt) in indices])
		return batch_idx, tgt_idx

	def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
		loss_map = {
			'labels': self.loss_labels,
			# 'cardinality': self.loss_cardinality,
			'boxes': self.loss_boxes,
			# 'masks': self.loss_masks
		}
		assert loss in loss_map, f'do you really want to compute {loss} loss?'
		return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


	def forward(self, outputs, targets):
		""" This performs the loss computation.
		Parameters:
			 outputs: dict of tensors, see the output specification of the model for the format
			 targets: list of dicts, such that len(targets) == batch_size.
					  The expected keys in each dict depends on the losses applied, see each loss' doc
		"""
		outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

		# Retrieve the matching between the outputs of the last layer and the targets
		indices = self.matcher(outputs_without_aux, targets)
		print

		# Compute the average number of target boxes accross all nodes, for normalization purposes
		num_boxes = sum(len(t["labels"]) for t in targets)
		num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
		
		torch.distributed.all_reduce(num_boxes)
		num_boxes = torch.clamp(num_boxes / torch.distributed.get_world_size(), min=1).item()

		# Compute all the requested losses
		losses = {}
		for loss in self.losses:
			losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

		return losses



def loss_functions(nc, phase='train'):
	
	matcher = build_matcher()
	losses = ['labels', 'boxes']
	weight_dict = {'loss_ce':2, 'loss_bbox':3, 'loss_giou':5}
	criterion = SetCriterion(nc, matcher=matcher, losses=losses, weight_dict=weight_dict, eos_coef=0.05).to(torch.distributed.get_rank())
	return criterion, matcher

