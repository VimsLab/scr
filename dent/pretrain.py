import os
import re
import math
import torch
import pickle
import random
import numpy as np
import torch.nn as nn
import concurrent.futures
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
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from transformer import Encoder


TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

from utils.plots import plot_images

def setup(rank, world_size):
    # Initialize the process group
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:28456",
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


class SiameseDataset(Dataset):
    def __init__(self, root, rank, phase, transform=None, apply_limit= False, lim=100, percent=1.0, num_workers=8):
        """
        lim: Upper limit of negative examples per image
        """

        self.file_list = glob(os.path.join(root, phase + '/*.pkl'))
        # self.count()
        # print(A)
        random.shuffle(self.file_list)
        self.transform = transform

        pos_file = phase+'positive.pkl'
        neg_file = phase+'negative.pkl'

        self.neg_limit = int(lim/num_workers)

        if not (Path(pos_file).is_file() and Path(neg_file).is_file()):
            if rank==0:
                print(f"\nCreating positve and negative pairs for {phase} since they don't exist yet")
            
            self.positive_pairs = []
            self.negative_pairs = []

            # Create pairs
            if rank ==0:
                print(('\n' + '%22s' * 3) % ('Image', 'Positive', 'Negative'))
            
            pbars = self.get_pbar(n=num_workers)
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                # submit tasks to the executor and get futures
                future1 = executor.submit(self.populate, pbars[0])
                future2 = executor.submit(self.populate, pbars[1])
                future3 = executor.submit(self.populate, pbars[2])
                future4 = executor.submit(self.populate, pbars[3])

                future5 = executor.submit(self.populate, pbars[4])
                future6 = executor.submit(self.populate, pbars[5])
                future7 = executor.submit(self.populate, pbars[6])
                future8 = executor.submit(self.populate, pbars[7])



                # wait for all tasks to complete and get results
                results = [future1.result(), future2.result(), future3.result(), future4.result(), future5.result(), future6.result()]            

            # self.negative_pairs = random.sample(self.negative_pairs, min(len(self.negative_pairs), int(2*len(self.positive_pairs))))
            if rank==0:
                print(len(self.positive_pairs), len(self.negative_pairs))
            with open(pos_file, 'wb') as f:
                pickle.dump(self.positive_pairs, f)

            with open(neg_file, 'wb') as f:
                pickle.dump(self.negative_pairs, f)

        else:
            if rank ==0:
                print(f"\nLoading positive and negative pairs from pickled lists of {phase}")
            with open(pos_file, 'rb') as f:
                self.positive_pairs = pickle.load(f)
                random.shuffle(self.positive_pairs)

            with open(neg_file, 'rb') as f:
                self.negative_pairs = pickle.load(f)
                random.shuffle(self.negative_pairs)

        if apply_limit:
            self.limiter(n=lim, save=neg_file)


        self.positive_pairs = random.sample(self.positive_pairs, int(len(self.positive_pairs)*percent)) 
        self.negative_pairs = random.sample(self.negative_pairs, int(len(self.negative_pairs)*percent))
        self.all_pairs = self.positive_pairs + self.negative_pairs
        random.shuffle(self.all_pairs)

        if rank==0:
            print(f'{phase} dataset has {len(self.positive_pairs)} positive pairs and {len(self.negative_pairs)} Negative pairs.')
            print(f'Ratio of negative to positive samples = {len(self.negative_pairs)/len(self.positive_pairs)}')

       

    def limiter(self, n=100, save=False):
        # selects 'n' companions for every unique member of self.negative pairs 
        # pickles the selected negative pair list if save=True

        # getting unique ids 
        t = [t[0].split('/')[-1] for t in self.negative_pairs]
        t1 = [t[1].split('/')[-1] for t in self.negative_pairs]
        t.extend(t1)
        unq = list(set(t))

        # dict with counts
        count = {k:0 for k in unq}
        new_negative_pairs = []
        for np in self.negative_pairs:
            aa, bb = np
            a, b = aa.split('/')[-1], bb.split('/')[-1]
            if count[a]<n and count[b]<n:
                new_negative_pairs.append((aa,bb))
                count[a] += 1
                count[b] += 1
        
        print(len(new_negative_pairs))
        print(new_negative_pairs[:5])
        if save:
            with open('new'+save, 'wb') as f:
                pickle.dump(new_negative_pairs, f)



    def get_pbar(self, n=4):
        size = math.ceil(len(self.file_list) / n)
        parts = [self.file_list[i:i+size] for i in range(0, len(self.file_list), size)]
        return parts


    def populate(self, flist):
        pbar = tqdm(enumerate(flist), total=len(flist), bar_format=TQDM_BAR_FORMAT)

        for i, fn_i in pbar:
            pid, tid, bid = self._get_person_id(fn_i)
            pos = 0
            neg = 0

            for j, fn_j in enumerate(flist[i+1:]):
                pjd, tjd, bjd = self._get_person_id(fn_j)
                
                diff = abs(int(bid) - int(bjd))
                # if different person id, make negative examples
                if pid != pjd and diff>4 and neg<self.neg_limit:
                    self.negative_pairs.append((fn_i, fn_j))
                    neg+=1

                # if same person id and same test id, make positive example
                elif tid == tjd and diff<4:
                    self.positive_pairs.append((fn_i, fn_j))
                    pos+=1

                pbar.set_description(('%22s' * 3) % (f'{tid+bid}', pos, neg))

        result = 'Done'
        return result



    def _get_person_id(self, filename):
        # define the pattern using regular expressions eg: '104L20_5.pkl'
        # get the test id eg: '104L2'
        # get the bscan id eg: '5'
        # get person id eg: '104L'

        fn = filename.split('/')[-1]        
        lr = fn.find('L')
        if lr==-1:
            lr = fn.find('R')

        pid = fn[:lr]
        testid = fn[:lr+2]
        # bscanid = (fn.split('.')[0]).split('_')[-1]

        # print(fn, pid, testid)
        # print(A)

        return pid, testid

    def count(self):
        files = self.file_list
        p=[]
        t=[]
        for f in files:
            person, test = self._get_person_id(f)
            p.append(person)
            t.append(test)

        print(len(set(p)), len(set(t)))
        print(set(t))
        l=0
        r = 0
        for c in set(t):
            if 'L' in c:
                l+=1
            elif 'R' in c:
                r+=1
        print(l,r)




    def __getitem__(self, index):
        filename1, filename2 = self.all_pairs[index]

        with open(filename1, "rb") as f:
            data1 = pickle.load(f)

        with open(filename2, "rb") as f:
            data2 = pickle.load(f)

        images1 = Image.fromarray(data1["img"][0]).convert('RGB').resize((576, 256))
        images2 = Image.fromarray(data2["img"][0]).convert('RGB').resize((576, 256))

        # Augmentations
        if self.transform:
            images1 = self.transform(images1)
            images2 = self.transform(images2)

        # Set target based on whether the pair is positive or negative
        target = torch.tensor(1 if (filename1, filename2) in self.positive_pairs else 0)

        return images1, images2, filename1, filename2, target


    def __len__(self):
        return len(self.all_pairs)


class SiameseNetwork(nn.Module):
    def __init__(self, encoder):
        super(SiameseNetwork, self).__init__()
        self.encoder = encoder
        
    def forward(self, x1, x2):
        embedding1 = self.encoder(x1)
        embedding2 = self.encoder(x2)
        # return torch.mean(embedding1, dim=1), torch.mean(embedding2, dim=1)
        return embedding1, embedding2
    
def get_loss(embeddings1, embeddings2, targets, margin=2.0):
        distances = F.pairwise_distance(embeddings1, embeddings2)
        loss = torch.mean((1-targets) * torch.pow(distances, 2) + targets * torch.pow(torch.clamp(margin - distances, min=0.0), 2))
        return loss

def contrastive_loss_cosine(embedding1, embedding2, similarity_label, margin=0.5):
    cosine_distance = 1 - F.cosine_similarity(embedding1, embedding2)
    loss = (1 - similarity_label) * 0.5 * cosine_distance**2 + similarity_label * 0.5 * torch.clamp(margin - cosine_distance, min=0)**2
    return loss.mean()

def cosineembeddingloss(e1, e2, targets, scale=2, margin=0.2):
    # target from -1 to 1
    target = (2*targets)-1 
    criteria = nn.CosineEmbeddingLoss(margin=margin, reduction='none')
    loss = criteria(e1,e2,target)
    return loss


# def contrastive_focal_loss(rank, emb1, emb2, target, margin=0.7, gamma=2, scale=2, eps=1e-2):
#     alpha = 1.5*torch.ones(1).to(emb1.device)
#     distance = F.pairwise_distance(emb1, emb2, p=2, keepdim=True).squeeze(-1) 
#     bce_loss = F.binary_cross_entropy_with_logits(distance.unsqueeze(0), target.unsqueeze(0).float(), reduction='none')
#     pt = torch.exp(-bce_loss)
#     # Compute the contrastive focal loss
#     cf_loss = (alpha*(1 - pt) ** gamma * bce_loss).squeeze(0)
#     cos_loss = cosineembeddingloss(emb1, emb2, target, margin=-1.0)    
#     cos_dist = 1 - F.cosine_similarity(emb1, emb2,dim=-1)
#     # print(distance.shape, cf_loss.shape, cos_dist.shape, cos_loss.shape)

#     if rank==0:
#         for t,d,cf,cd,cl in zip(target, distance, cf_loss, cos_dist, cos_loss):
#             print('\nTarget:', t.item(), 'l1 distance:', d.item(), 'contrastive loss:', cf.item())
#             print('Cosine distance:', cd.item(), 'Cosine loss:', cl.item())
    
#     loss = cf_loss + cos_loss
#     loss = torch.mean(loss)

#     print('\n')

#     return loss


def contrastive_focal_loss(rank, emb1, emb2, target, margin=0.7, gamma=2, scale=2, eps=1e-5, alpha=0.7):
    
    distance = F.pairwise_distance(emb1, emb2, p=2, keepdim=True).squeeze(-1).sigmoid()

    distance = torch.clip(distance, eps, 1-eps)

    p_t = torch.where(target==1, 1-distance, distance)

    alpha = torch.ones_like(target)*alpha

    alpha_t = torch.where(target==1, alpha, 1-alpha)

    ce = -torch.log(p_t)

    cf_loss = ce * alpha_t * (1-p_t)**gamma


    cos_loss = cosineembeddingloss(emb1, emb2, target, margin=-1.0)    
    cos_dist = 1 - F.cosine_similarity(emb1, emb2,dim=-1)

    if rank==0:
        for t,d,cf,cd,cl in zip(target, distance, cf_loss, cos_dist, cos_loss):
            print('\nTarget:', t.item(), 'l1 distance:', d.item(), 'contrastive loss:', cf.item())
            print('Cosine distance:', cd.item(), 'Cosine loss:', cl.item())
    
    loss = 0.8*cf_loss + 0.2*cos_loss
    loss = torch.mean(loss)

    print('\n')

    return loss


def train_epoch(rank, siamese_net, optimizer, train_loader, epoch, epochs, running_loss=0):  
    siamese_net.train()

    if rank ==0:
        print(('\n' + '%22s' * 6) % ('Device', 'Epoch', 'GPU Mem', 'E1minmax', 'E2minmax','Loss'))

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:+10b}')
    # with torch.autograd.detect_anomaly():
    for batch_idx, (x1, x2, f1, f2, targets) in pbar:
        # print(f1, f2, targets)
        x1 = x1.to(rank, non_blocking=True)
        x2 = x2.to(rank, non_blocking=True)
        targets = targets.to(rank, non_blocking=True)
        optimizer.zero_grad()
        
        # Forward pass
        with autocast():
            embeddings1, embeddings2 = siamese_net(x1, x2)
            loss = contrastive_focal_loss(rank, embeddings1[:,-1], embeddings2[:,-1], targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        running_loss = running_loss+loss.item()


        if rank==0:
            # try:
            # print statistics
            minmaxe1 = f'({torch.min(embeddings1).item():.4g}, {torch.max(embeddings1).item():.4g})'
            minmaxe2 = f'({torch.min(embeddings2).item():.4g}, {torch.max(embeddings2).item():.4g})'
            mem = f'{torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%22s'*5 + '%22.4g') % 
                (f'{rank}', f'{epoch}/{epochs - 1}', mem, minmaxe1, minmaxe2, running_loss/(batch_idx+1)))
            # except:
            #     pass
            
    return siamese_net



def validate(rank, siamese_net, val_loader, thres=0.1):
    siamese_net.eval()
    with torch.no_grad():
        total_loss = 0 
        corrects = 0
        tps = 0
        tns = 0
        total = 0

        if rank==0:
            print(('\n' + '%22s' * 5) % ('Correct', '(TP,P)', '(TN,N)', 'Accuracy', 'Loss'))
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        for batch_idx, (x1, x2, f1, f2, targets) in pbar:
            x1 = x1.to(rank, non_blocking=True)
            x2 = x2.to(rank, non_blocking=True)
            targets = targets.to(rank, non_blocking=True)

            # Forward pass
            embeddings1, embeddings2 = siamese_net(x1, x2)

            loss = contrastive_focal_loss(rank, embeddings1[:,-1], embeddings2[:,-1], targets)

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

            if rank==0:
                pbar.set_description(('%22s'*3 +'%22.4g' * 2) % (correct, f'({tp},{p})', f'({tn},{n})', correct/total, loss.item()))


        # calculate average loss and accuracy
        avg_loss = total_loss / len(val_loader)
        accuracy = corrects / total
    if rank==0:
        print(('\n'+ '%22s') % ('Validation stats:'))
        print(('%22s' * 6) % ('Total', 'TP', 'TN', 'Incorrect', 'avg_acc', 'avg_loss'))
        print(('%22s' * 4 + "%22.4g"*2) % (total, f'{tps}/{corrects}', f'{tns}/{corrects}', total-corrects, accuracy, avg_loss))

    return avg_loss, accuracy


def tx():
    tx_dict = {'train':transforms.Compose([
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),

        'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        }
    return tx_dict


def get_dataset(world_size, rank, dataroot, phase, lim, transform, apply_limit=False, batch_size=64, percent=1.0, shuffle=False, num_workers=8):

    dataset = SiameseDataset(dataroot, rank, phase, transform, apply_limit=apply_limit, lim=lim, percent=percent)
    if world_size>0:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    else:
        sampler=None

    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return dataloader, sampler



def pretrainer(rank, world_size, root, dataroot, phases=['sample', 'sample'], resume=False):
    setup(rank, world_size)

    num_epochs = 152
    batch_size = 64 #// world_size
    
    tx_dict = tx()
    train_loader, train_sampler = get_dataset(world_size, rank, dataroot, 
                                            phase=phases[0], lim=100, 
                                            transform=tx_dict['train'], 
                                            batch_size=batch_size, percent=1)
    val_loader, val_sampler = get_dataset(world_size, rank, dataroot, 
                                        phase=phases[1], lim=8, apply_limit=False,
                                        transform=tx_dict['val'], 
                                        batch_size=batch_size)

    # create model and optimizer
    encoder = Encoder(hidden_dim=256, num_encoder_layers=6, nheads=8)

    siamese_net = SiameseNetwork(encoder).to(rank)


    # Wrap the model with DistributedDataParallel
    siamese_net = DDP(siamese_net, device_ids=[rank], find_unused_parameters=False)

    optimizer = torch.optim.Adam(siamese_net.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_accuracy = 0
    start_epoch = 0

    if resume:
        ckptfile = root + resume + '.pth'
        ckpts = torch.load(ckptfile, map_location='cpu')
        siamese_net.load_state_dict(ckpts['model_state_dict'])
        optimizer.load_state_dict(ckpts['optimizer_state_dict'])
        start_epoch = ckpts['epoch']+1
        best_accuracy = ckpts['best_val_acc']

        if rank == 0:
            print('Resuming training from epoch {}. Loaded weights from {}. Last best accuracy was {}'
                .format(start_epoch, ckptfile, best_accuracy))


    # Train the network
    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)
        siamese_net = train_epoch(rank, siamese_net, optimizer, train_loader, epoch, num_epochs, running_loss=0)
        
        # Update the learning rate
        lr_scheduler.step()

        vloss, acc = validate(rank, siamese_net, val_loader)

        # torch.distributed.barrier()
        if rank==0:
            if acc>=best_accuracy:
                best_accuracy = acc
                # save_path = root + 'epoch' + str(epoch) + 'best_pretrainer.pth'
                save_path = root + 'best_pretrainer.pth'
            else:
                save_path = root + 'last_pretrainer.pth'
            
            checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': siamese_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_accuracy,
                }
            torch.save(checkpoint, save_path)


    # Clean up the process group
    cleanup()            


__all__ = ['pretrainer', 'train_epoch', 'SiameseDataset', 'Encoder', 'SiameseNetwork', 'contrastive_loss_cosine', 'contrastive_focal_loss',
           'validate', 'tx', 'get_dataset', 'setup', 'cleanup']

if __name__ == '__main__':

    root = '/media/jakep/eye/scr/dent/'
    dataroot = '../pickle/'


    # train_loader, train_sampler = get_dataset(0, 0, dataroot, 
    #                                         phase='train2', lim=100, 
    #                                         batch_size=1, percent=0.01, transform=None)
    val_loader, val_sampler = get_dataset(0, 0, dataroot, 
                                        phase='val2', lim=8, apply_limit=False,
                                        batch_size=1, transform=None)

    # ddp
    world_size = 2
    mp.spawn(pretrainer, args=(world_size, root, dataroot, ['train2', 'val2'], False), nprocs=world_size, join=True)
