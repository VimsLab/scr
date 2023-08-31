import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from glob import glob
import numpy as np 
import os
from PIL import Image
import cv2 as cv
import shutil
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from os import path 
import seaborn as sns
import pickle
import torch
import torchvision.transforms.v2 as transforms
import torch.nn.functional as F
from natsort import natsorted


main_folder = "../pickle_pretrain"
# sub_folders = os.listdir(main_folder)
# sub_folders = [path.join(main_folder, sf) for sf in sub_folders if path.isdir(path.join(main_folder, sf))]
# ann_folder = "original_annotations"


data_dir = 'data'
pretrain_data_dir = path.join(data_dir, 'pretrain')
detector_data_dir = path.join(data_dir, 'detector')
data_file_name = 'fold'
folds = 5




def plot_attention(image, attention_matrix, name):
	# Generate a random attention matrix
	# attention_matrix = np.random.rand(16, 16)

	for i, a, n in zip(image, attention_matrix, name):
		plot_one_attention(i,a,n)

   


def plot_one_attention(image, attention_matrix, name):

	fn, p, t = id_per_file(name)
	_, h,w = image.shape
	
	image = image.permute(1,2,0).cpu() #* 0.5 + 0.5
	# print(torch.max(attention_matrix), torch.min(attention_matrix))
	atnmat = torch.max(attention_matrix.cpu(), dim=-1)
	attention_matrix = atnmat[0]
	m = int(np.sqrt(attention_matrix.shape))

	attention_matrix = F.interpolate(attention_matrix.view(1, 1, m, m), (h, w), mode='bilinear').view(h, w)

	plt.imshow(image)

	# alphas = np.clip(attention_matrix,0,1)*0.5

	# ax = sns.heatmap(attention_matrix, alpha=0.6, cmap=sns.color_palette("WOrR", as_cmap=True))
	plt.imshow(attention_matrix, alpha=0.6, cmap='Blues')

	plt.axis('off')
	# Save the figure
	plt.savefig('imagetest/'+ fn + '.png')
	plt.close()



def ids(filename):
	'''
	Get person id and test id
	'''
	f, p, t = [], [], []

	for files in filename:
		file, pid, testid = id_per_file(files)
		f.append(file)
		p.append(pid)
		t.append(testid)
	return f, p, t


def id_per_file(fn):
	file = (fn.split('/')[-1]).split('.')[0]
	lr = file.find('L')
	if lr==-1:
		lr = file.find('R')
	pid = file[:lr+1]
	testid = file[:lr+3]
	return file, pid, testid

def root_tail(fn):
	file = fn.replace('.pkl', '')
	tail = file.split('_')[-1]
	root = file[:-len(tail)-1]

	return root, int(tail)


def boxes_cxcywh_to_xyxy(v):
	boxes = []
	for val in v:
		boxes.append(box_cxcywh_to_xyxy(val))
	return np.stack(boxes)


def box_cxcywh_to_xyxy(v):
	if type(v)==torch.Tensor:
		x_c, y_c, w, h = v.unbind(-1)
	else:
		x_c, y_c, w, h = v
	b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
	return torch.tensor(b)


def pltbbox(image, bbox=None, cls=None, name='a', clr='r', im_save_dir='imagetest/', score=None, th=0.7):
	colors = ['cyan', 'orange']
	classes = ['FOVEA', 'SCR']

	if type(image)==str:
		image = plt.imread(image)

	if type(image)==torch.Tensor:
		_,h, w = image.shape
		image = image.permute(1,2,0)

	elif isinstance(image, Image.Image):
		w, h = image.size

	plt.imshow(image, aspect=1)
	conf=None

	if score is not None:
		conf=score[score>th]
		if len(conf)<1:
			bbox=None
			cls=None
		else:
			bbox=bbox[score>th]
			cls=cls[score>th]

	if bbox is not None and len(bbox)>0:
		bbox = boxes_cxcywh_to_xyxy(bbox)
		
		for i, box in enumerate(bbox):
			lab = (cls[i]-1).long().item()
			a,b,c,d = box
			plt.gca().add_patch(Rectangle((w*a, h*b), width=w*(c-a), height=h*(d-b), edgecolor=colors[lab], facecolor='none'))
			if cls is not None:
				
				if conf is not None:
					string = classes[lab] + ' '+('%.2f')%(conf[lab])

				else:
					string = classes[lab]

				plt.annotate(string, (w*a + w*(c-a)/2, h*b + h*(d-b)/4), color='white', weight='normal', fontsize=8, ha='center', va='top')
	if name is not None:
		name = path.join(im_save_dir, name)
		plt.gca().set_axis_off()
		plt.savefig(name+'.png', bbox_inches = 'tight', pad_inches = 0)
		plt.close()


def get_pickles(sff=main_folder):
	pkls = natsorted(glob(path.join(sff) + '/*.pkl'))
	return pkls


def save_pickle(im, bx, cx, n):
	fn = path.join(pickle_dir, n)
	tensor = {'img':im, 'box':bx, 'label':cx, 'name':n}
	with open(fn + '.pkl', 'wb') as f:
		pickle.dump(tensor, f)
	

def load_pickles():
	pick_list = []
	pick = natsorted(glob(pickle_dir + '/*.pkl'))
	for p in pick:
		pick_list.append(load_one_pickle(p))
	return pick_list


def load_one_pickle(fn):
	with open(fn, 'rb') as f:
		pic = pickle.load(f)
	return pic


def operation_and_save(i, an, fnlist, n, plot=True, save=True):
	c = torch.tensor(an[:,-1]).int()
	a = an[:,:-1]
	fname = path.join(ann_save_dir, n)
	for f in fnlist:
		if f == 'crop':
			i, a = make_crop(i, a)

		if f == 'tx':
			i, a = transform(i, a)

	if c[0]==2:
		a = make_fake_annotation()[:,:-1]	

	if plot and c[0]!=2:
		pltbbox(i, a.numpy(), cls=c.numpy(), name=n)
		if save:
			write_annotation(a, c, fname)

	if save:
		save_pickle(i, a.to(torch.float16), c, n)
	


def make_images_and_annos(imf, anf, f, fxn=['crop', 'tx'] ):
	for i, a, n in zip(imf, anf, f):
		i = read_one_image(i)
		a = read_one_annotation(a)
		print(n)
		operation_and_save(i, a, fxn, n=n)



def get_positives(tidgroup):
	positives = []
	for k, v in tidgroup.items():
		pairs = [(v[i], v[j], 1) for i in range(len(v)) for j in range(i+1, min(i+4, len(v))) ]
		positives.extend(pairs)
	print('positives length=', len(positives))
	return positives


def get_negatives(pg):
	negatives = []
	keys = list(pg.keys())
	l = len(keys) - 1
	for i, k in enumerate(keys):
		v = pg[k]
		corr = min(i+1, l-1)
		pair_k = random.choices(keys[corr:l], k=len(v)*20)
		pair_v = [random.sample(pg[pk],1)[0] for pk in pair_k ]
		
		pairs = [(x,y, 0) for x,y in zip(v*20, pair_v) ]
		
		negatives.extend(pairs)
	print('negative length=', len(negatives))
	return negatives


def write_to_file(name, val, pf, sf):
	with open(path.join(data_dir, name + '.txt'), 'w') as f:
		f.write(''.join([f'{pf + p[0] + sf} {pf + p[1] + sf} {p[2]}\n'for p in val]))


def read_from_file(fold, name, pretrain=True):
	if pretrain: 
		name = name + '_' + str(fold) + '.txt'
		values = []
		with open(path.join(data_dir, name), 'r') as f:
			for paths in f.readlines():
				try:
					a,b,c = paths.split(' ')
				except ValueError as e:
					continue
				values.append((a,b,int(c[:-1])))

	return values


def dataset_pretrainer(pg, tg,f):
	prefix = main_folder + '/'
	suffix = '.pkl'

	positives = get_positives(tg)
	write_to_file('positive_'+str(f), positives, prefix, suffix)

	negatives = get_negatives(pg)
	write_to_file('negative_'+str(f), negatives, prefix, suffix)

	print(len(positives), len(negatives))

	return positives, negatives



def create_groups(title, filenames, id):
	group = {k:[] for k in title}
	for fn, i in zip(filenames, id):
		group[i].append(fn)

	return group


def folding(pid, tid, fold=3):
	
	q, rem = divmod(len(pid), fold)

	fold_pid = []
	fold_tid = []

	for i in range(fold):
		r = rem*(i==fold-1)
		this_fold = pid[i*q: i*q+q+r]

		this_test = [t for t in tid if t[:-2] in this_fold]

		fold_pid.append(this_fold)
		fold_tid.append(this_test)

	return fold_pid, fold_tid


def fold_operation(f,p,t, fold):
	unique_pid = list(set(p))
	unique_tid = list(set(t))

	group_pid = create_groups(unique_pid, f, p)
	group_tid = create_groups(unique_tid, f, t)

	fp, ft = folding(unique_pid, unique_tid, fold)

	for i in range(fold):
		subkeysp = fp[i]
		p = {a:group_pid[a] for a in subkeysp}
		
		subkeyst = ft[i]
		t = {a:group_tid[a] for a in subkeyst}
		pos, neg = dataset_pretrainer(p, t,i)


def split(fold, itr=0):
	train = []
	val = []
	# print('\n\n')
	for f in range(fold):
		pos = read_from_file(f, 'positive')
		neg = read_from_file(f, 'negative')
		data = pos + neg 
		# print(len(pos), len(neg))
		random.shuffle(data)
		if f == itr:
			# print('VAL*************\n')
			val.extend(data)
		else:
			# print('TRAIN*********\n')
			train = train + data
	return train, val


def train_val_split(name, pid, group):

	prefix = main_folder + '/'
	suffix = '.pkl'

	data = [v for k,v in group.items() if k in pid]
	data = [a for b in data for a in b]

	with open(path.join(data_dir, name + '.txt'), 'w') as f:
		f.write(''.join([f'{prefix + p + suffix}\n'for p in data]))
	return data


def read_data(name):
	data = []
	with open(path.join(data_dir, name + '.txt'), 'r') as f:
		for paths in f.readlines(): 
			data.append(paths.replace('\n', ''))

	return data

def dataset_trainer(f, p, t):
	unique_pid = list(set(p))
	pid_group = create_groups(unique_pid, f, p)
	th = int(0.8*len(unique_pid))
	train_pid, val_pid = unique_pid[0:th], unique_pid[th:]

	train = train_val_split('train', train_pid, pid_group)
	val = train_val_split('val', val_pid, pid_group)



if __name__ == '__main__':


	pkl_files = get_pickles(main_folder)
	random.shuffle(pkl_files)
	f, p, t = ids(pkl_files)
	
	print(len(f), len(p), len(t))

	fold_operation(f,p,t,folds)

	t, v = split(folds, 0)
	print(len(t), len(v))

	# dataset_trainer(f,p,t)
	# read_data('val')
	


