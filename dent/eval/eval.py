import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from loss.matching_loss import box_cxcywh_to_xyxy, generalized_box_iou
from .metrices import box_iou, fitness, ap_per_class, CF
from utils.plots import plot_images, output_to_target
import wandb


class Meter:
	def __init__(self, elem, rank):
		self.val = torch.empty((1, elem)).to(rank)
		self.count = 0 
		self.mean = torch.zeros((1, elem)).to(rank)
		self.sum = torch.zeros((1, elem)).to(rank)
		self.rank = rank

	def adds(self, value, dim=0):
		value = value.to(self.rank)
		if value.dim()<2:
			value = value.reshape((1,-1))

		self.val = torch.cat([self.val, value], dim)
		if self.count == 0:
			self.val = self.val[1:]

		self.count += 1
		return self.val

	def sums(self, dim=0):
		if dim=='r':
			dim = [i for i in range(self.val.dim())]
		
		# torch.distributed.all_reduce(self.val)
		self.sum = torch.round(torch.sum(self.val, dim), decimals=3)
		return self.sum

	def means(self, dim=0):
		if dim=='r':
			dim = [i for i in range(self.val.dim())]

		# torch.distributed.all_reduce(self.val)
		self.mean = torch.round(torch.mean(self.val, dim), decimals=3)
		return self.mean

	def returns(self, param):
		return param.detach().cpu().numpy().tolist()




def process_batch(detections, labels, iouv):
	"""
	Return correct prediction matrix
	Arguments:
		detections (array[N, 6]), x1, y1, x2, y2, conf, class
		labels (array[M, 5]), class, x1, y1, x2, y2
	Returns:
		correct (array[N, 10]), for 10 IoU levels
	"""
	correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
	iou = box_iou(labels[:, 1:], detections[:, :4])
	
	correct_class = labels[:, 0:1] == detections[:, 5]

	for i in range(len(iouv)):
		x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
		if x[0].shape[0]:
			matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
			if x[0].shape[0] > 1:
				matches = matches[matches[:, 2].argsort()[::-1]]
				matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
				matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
			correct[matches[:, 1].astype(int), i] = True
	
	# print('SUM', correct.sum())
	return torch.tensor(correct, dtype=torch.bool) #, device=iouv)


class EvaluationCriterion():
	def __init__(self, rank, plot=False,  nc=2, conf_thres=0.6, iou_thres_nms=0.85, iou_thres_lb=0.7):
		self.rank = rank
		
		self.conf_thres = conf_thres
		self.iou_thres = iou_thres_nms
		self.nc = nc
		self.iouv = torch.linspace(0.5, 0.95, 1, device=rank)
		self.cf = [CF(nc=nc)]*len(self.iouv)
		self.plot = plot
	

	def make_targets(self, targets):
		target = []
		for i, t in enumerate(targets):
			if len(t)>0:
				labels, boxes = t[:,0], t[:,1:]
				tgt = torch.zeros((len(t),6))
				tgt[:,0] = i 
				tgt[:, 1] = labels 
				tgt[:, 2:] = box_cxcywh_to_xyxy(boxes)
				target.append(tgt)
		if len(target)<1:
			target = torch.empty((0,6))

		else:
			target = torch.concatenate(target, dim=0)
		return target



	def make_preds(self, preds, targets):
		pred = []
		l, b = preds['pred_logits'], preds['pred_boxes']
		nb = l.shape[0]
		
		for i, (labels, boxes) in enumerate(zip(l,b)):
			conf, lb = torch.max(labels, dim=-1)
			ped = torch.zeros((len(labels),6))
			ped[:, 0:4] = boxes #box_cxcywh_to_xyxy(boxes) 
			ped[:, 4] = conf
			
			ped[:,5] = lb
			pred.append(ped)

		pred = torch.stack(pred, dim=0)
		
		lb = [targets[targets[:,0] == i, 1:] for i in range(nb)]
		pred, pr = non_max_suppression(pred, self.conf_thres, self.iou_thres, nc=2)
		# self.prlog.append(pr)
		return pred


	@torch.no_grad()
	def evalcriterion(self, batch, images, targets, preds, pths, save_dir, names=['Fovea', 'SCR'], phase='val'):
		targets = self.make_targets(targets)
		preds = self.make_preds(preds, targets)

		if self.rank==0 and self.plot and batch<20:
			plot_images(images, targets, pths, save_dir + phase + f'_batch{batch}_labels.jpg', names)  # labels
			plot_images(images, output_to_target(preds), pths, save_dir + phase + f'_batch{batch}_pred.jpg', names)

		for si, pred in enumerate(preds):
			labels = targets[targets[:,0] == si, 1:]
			nl, npr = labels.shape[0], pred.shape[0]
			niou = self.iouv.numel() 
			device = labels.device
			correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)
			
			if npr == 0:
				if nl:
					# self.stat.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
					bg_class = [ (labels[:,0]==1).sum(), (labels[:,0]==2).sum(), 0]
					for i, c in enumerate(self.cf):
						c.process_batch(bg_class, 'bglab')
			
			else:
				x = pred[:,5]==1
				y = pred[:,5]==2

				# self.nprs_f += x.sum()
				# self.nprs_s += y.sum()

				if nl==0:
					bg_class = [x.sum(), y.sum(), 0]
					for c in self.cf:
						c.process_batch(bg_class, 'bgpred')

				elif nl and npr:
					correct = process_batch(pred.cpu(), labels.cpu(), self.iouv.cpu())
					correct = correct.to(device)
					
					for i, crr in enumerate(correct.clone().T):
						# print(i, (crr[x]).sum(), (crr[y]).sum(), (~crr[x]).sum(), (~crr[y]).sum(), crr.shape, correct.shape, correct.clone().T.shape)
						self.cf[i].process_batch([(crr[x]).sum(), (crr[y]).sum(), (~crr[x]).sum(), (~crr[y]).sum()], 'tpfp')
			
			
		

	def calcmetric(self, plot=False, save_dir='', names=['Fovea', 'SCR'], loss=None):
		if self.plot:
			self.cf[0].endloop()
			self.cf[0].plot(save_dir=save_dir, names=names)
		# print('\n0 tp and fp and fn:',self.cf[0].tp_fp())
		print('matrix:\n',self.cf[0].nmatrix)
		p = []
		r = []
		for i, c in enumerate(self.cf):
			tp, fp, fn, _ = c.tp_fp()
			precision = np.divide(tp, np.add(tp, fp)+0.001)
			recall = np.divide(tp, np.add(tp, fn)+0.001)
			p.append(precision)
			r.append(recall)
			# print('\n',self.iouv[i].cpu().item(), tp, fp, fn, precision, recall)

		# f = np.stack([p[0], r[0], p[-1], r[-1]]).reshape((self.nc,-1))
		# f = fitness(f)
		tp, fp, fn, tn = c.tp_fp()
		f=[]
		for i, n in enumerate(names):
			tpp = tp[i].item()
			fpp = fp[i].item()
			fnn = fn[i].item()
			tnn = tn[i].item()

			f.append((tpp+tnn)/(1+tpp+fpp+fnn+tnn))

			print('\n', n, 'TP FP FN TN:', [tpp, fpp, fnn, tnn])

		f = np.array(f)
		p = np.array(p)
		r = np.array(r)
		if self.rank==0:
			print(('\n%44s' + '%11s' * 5) %('Labels', 'P', 'R', 'P@0.5', 'P@0.9', 'Fitness'))
			for i in range(self.nc):
				print(('%44s'  + '%11.3g' * 5) %(names[i], np.mean(p[:,i]), np.mean(r[:,i]), p[0][i], p[-1][i], f[i]))

		f=f.mean()
		if self.rank==0:
			print(('%44s' + '%11.3g' * 6) %('Total', np.mean(p), np.mean(r), np.mean(p[0]), np.mean(p[-1]), f, loss))		

		return f




def non_max_suppression(prediction, conf_thres, iou_thres, nc=2, max_det=5):
	device = prediction.device 
	bs = prediction.shape[0]
	pr = []


	assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
	assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

	xc = prediction[..., 4] > conf_thres 

	output = [torch.zeros((0, 6), device=device)] * bs
	for xi, x in enumerate(prediction):  # image index, image inference
		x = x[xc[xi]]  # confidence

		if len(x):
			x[:,:4] = box_cxcywh_to_xyxy(x[:,:4])			
			z = torchvision.ops.nms(x[...,:4], x[...,4], iou_thres)

			v = x[z]

			ff = torch.where(v[...,5]==1)[0].cpu().numpy().tolist()
			gg = torch.where(v[...,5]!=1)[0].cpu().numpy().tolist()

			if len(ff)>1:
				gg.append(ff[0])
				gg.sort()

				v = v[gg]
				# v = torch.cat([v[:ff[1]], v[ff[1]:]])
		
			if len(v)> max_det:
				v = v[:max_det]

			output[xi] = v

			# pr.append(v[...,4])
	return output, None


