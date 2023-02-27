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
from torch.nn.parallel import DataParallel
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

def setup(rank, world_size):
    # Initialize the process group
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:24456",
        rank=rank,
        world_size=world_size
    )
    # Set the GPU to use
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()



class SiameseDataset(Dataset):
    def __init__(self, root, rank, phase, transform=None, apply_limit= False, lim=100, num_workers=4):
        """
        lim: Upper limit of negative examples per image
        """

        self.file_list = glob(os.path.join(root, phase + '/*.pkl'))
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
            
            pbars = self.get_pbar(n=4)

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # submit tasks to the executor and get futures
                future1 = executor.submit(self.populate, pbars[0])
                future2 = executor.submit(self.populate, pbars[1])
                future3 = executor.submit(self.populate, pbars[2])
                future4 = executor.submit(self.populate, pbars[3])


                # wait for all tasks to complete and get results
                results = [future1.result(), future2.result(), future3.result(), future4.result()]            

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

        if rank==0:
            print(f'{phase} dataset has {len(self.positive_pairs)} positive pairs and {len(self.negative_pairs)} Negative pairs.')
            print(f'Ratio of negative to positive samples = {len(self.negative_pairs)/len(self.positive_pairs)}')

        if apply_limit:
            self.limiter(n=lim, save=neg_file)


        self.all_pairs = self.positive_pairs + self.negative_pairs
        random.shuffle(self.all_pairs)
    

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
        lr = fn.find('L')+1
        if not lr:
            lr = fn.find('R')+1

        pid = fn[:lr]
        testid = fn[:lr+1]
        bscanid = (fn.split('.')[0]).split('_')[-1]

        return pid, testid, bscanid



    def __getitem__(self, index):
        filename1, filename2 = self.all_pairs[index]

        with open(filename1, "rb") as f:
            data1 = pickle.load(f)

        with open(filename2, "rb") as f:
            data2 = pickle.load(f)

        images1 = Image.fromarray(data1["img"][0]).convert('RGB').resize((256, 224))
        images2 = Image.fromarray(data2["img"][0]).convert('RGB').resize((256, 224))

        # Augmentations
        if self.transform:
            images1 = self.transform(images1)
            images2 = self.transform(images2)

        # Set target based on whether the pair is positive or negative
        target = torch.tensor(1 if (filename1, filename2) in self.positive_pairs else 0)

        return images1, images2, filename1, filename2, target


    def __len__(self):
        return len(self.all_pairs)


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=0.1)
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, hidden_dim=256, nheads=8, num_encoder_layers=6, dropout_rate=0.1):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.num_encoder_layers = num_encoder_layers
        self.dropout_rate = dropout_rate
        
        # Create a positional encoding module
        self.position_embedding = nn.Parameter(torch.randn(1, hidden_dim, 1, 1))
        
        # Create a linear layer for embedding the encoder features
        self.linear_emb = nn.Linear(2048, hidden_dim)
        
        # Create a transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads, dropout=dropout_rate)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
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
        encoder_outputs = self.encoder(encoder_embedding.transpose(0, 1)) # shape: (seq_len, batch_size, hidden_dim)
        
        return encoder_outputs



# class DETREncoder(nn.Module):
#     def __init__(self, hidden_dim=256, num_layers=6, nhead=8):
#         super().__init__()
#         backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
#         self.backbone = nn.Sequential(*list(backbone.children())[:-2])
#         self.conv = nn.Conv2d(in_channels=2048, out_channels=hidden_dim, kernel_size=1)
#         self.pos_encoder = PositionalEncoding(hidden_dim)
#         self.transformer_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead),
#             num_layers=num_layers
#         )        
#         self.fc = nn.Linear(in_features=2048, out_features=hidden_dim)

#     def forward(self, x):
#         features = self.backbone(x)  # (batch_size, 2048, H/32, W/32)

#         # features = features.mean(dim=[2, 3])  # (batch_size, 2048)
#         # features = self.fc(features)  # (batch_size, hidden_dim)
#         # features = features.unsqueeze(1)  # (batch_size, 1, hidden_dim)
#         # features = features.unsqueeze(0)  # (1, batch_size, hidden_dim) ~ (num_pixels, batch_size, hidden_dim)
#         features = self.conv(features) # (batch, hidden_dim, H/32, W/32)
#         features = features.flatten(2).permute(2, 0, 1)  # shape: (num_pixels, batch_size, hidden_dim)
#         features = self.pos_encoder(features)  # shape: (num_pixels, batch_size, hidden_dim)
#         # Transformer encoder
#         encoded = self.transformer_encoder(features)  # shape: (num_pixels, batch_size, hidden_dim) 
#         # encoded = self.transformer_encoder(features)  # (batch_size, 1, hidden_dim)
#         # encoded = encoded.squeeze(1)  # (batch_size, hidden_dim)

#         # encoded = encoded.permute(1, 0, 2) # (batch_size, num_pixels, hidden_dim) 

#         return encoded


class SiameseNetwork(nn.Module):
    def __init__(self, encoder):
        super(SiameseNetwork, self).__init__()
        self.encoder = encoder
        
    def forward(self, x1, x2):
        embedding1 = self.encoder(x1)
        embedding2 = self.encoder(x2)
        return torch.mean(embedding1, dim=0), torch.mean(embedding2, dim=0)
    
def get_loss(embeddings1, embeddings2, targets, margin=2.0):
        distances = F.pairwise_distance(embeddings1, embeddings2)
        loss = torch.mean((1-targets) * torch.pow(distances, 2) + targets * torch.pow(torch.clamp(margin - distances, min=0.0), 2))
        return loss

def contrastive_loss_cosine(embedding1, embedding2, similarity_label, margin=0.5):
    cosine_distance = 1 - F.cosine_similarity(embedding1, embedding2)
    loss = (1 - similarity_label) * 0.5 * cosine_distance**2 + similarity_label * 0.5 * torch.clamp(margin - cosine_distance, min=0)**2
    return loss.mean()


def validate(rank, siamese_net, val_loader, thres=0.55):
    siamese_net.eval()
    with torch.no_grad():
        total_loss = 0 
        correct = 0
        total = 0

        print(('\n' + '%22s' * 4) % ('Device', 'Correct', 'Accuracy', 'Loss'))
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        for batch_idx, (x1, x2, f1, f2, targets) in pbar:
            x1 = x1.to(rank, non_blocking=True)
            x2 = x2.to(rank, non_blocking=True)
            targets = targets.to(rank, non_blocking=True)

            # Forward pass
            embeddings1, embeddings2 = siamese_net(x1, x2)
            loss = contrastive_loss_cosine(embeddings1, embeddings2, targets)

            # calculate accuracy
            pred = (torch.sign(embeddings1 - embeddings2) + 1) / 2  # convert to binary predictions
            emb_len = pred.shape[-1]
            score = torch.sum(pred,dim=-1)
            op = torch.zeros_like(score)
            op[score>emb_len*thres] =1

            correct += op.eq(targets).sum().item()
            total += targets.size(0)

            # accumulate loss
            total_loss += loss.item()

            pbar.set_description(('%22s'*2 +'%22.4g' * 2) % (f'{rank}', f'{correct}/{total}', correct/total, total_loss/(batch_idx+1)))


        # calculate average loss and accuracy
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

    return avg_loss, accuracy


def train_epoch(rank, siamese_net, optimizer, train_loader, epoch, epochs, running_loss=0):  
    siamese_net.train()
    prev_valid_loss = torch.zeros(1)

    if rank ==0:
        print(('\n' + '%22s' * 4) % ('Device', 'Epoch', 'GPU Mem', 'Loss'))

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:+10b}')
    # with torch.autograd.detect_anomaly():
    #     with torch.autograd.profiler.profile():
    for batch_idx, (x1, x2, f1, f2, targets) in pbar:
        # print(f1, f2, targets)
        x1 = x1.to(rank, non_blocking=True)
        x2 = x2.to(rank, non_blocking=True)
        targets = targets.to(rank, non_blocking=True)

        optimizer.zero_grad()
        
        # Forward pass
        with autocast():
            embeddings1, embeddings2 = siamese_net(x1, x2)
            loss = contrastive_loss_cosine(embeddings1, embeddings2, targets)

            if torch.any(torch.isnan(embeddings1)) or torch.any(torch.isnan(embeddings2)) or torch.isnan(loss):
                with open(str(batch_idx)+'ERRORLOG.txt', 'w+') as f:
                    for a in range(len(f1)):
                        f.write(f'[{f1[a]}, {f2[a]}]')
                    f.write(f'\nAny: {torch.any(torch.isnan(embeddings1))}, {torch.any(torch.isnan(embeddings2))},\
                        All: {torch.all(torch.isnan(embeddings1))}, {torch.all(torch.isnan(embeddings2))},\
                        Loss: {loss.item()}, {prev_valid_loss.item()}')
        
                loss = prev_valid_loss
                
                print('\nStopping early because NaN encountered. Previous loss was',prev_valid_loss.item())
                break

            else:
                prev_valid_loss = loss

            
        # Backward pass and optimization
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(siamese_net.parameters(), 1)
        # grads = [p.grad.detach().flatten() for p in siamese_net.parameters() if p.grad is not None]
        # print('\nafter clip', torch.max(grads), torch.min(grads))
        optimizer.step()
        
        running_loss = running_loss+loss.item()


        if rank==0:
            try:
                # print statistics
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%22s'*3 + '%22.4g' * 1) % (f'{rank}', f'{epoch}/{epochs - 1}', mem, running_loss/(batch_idx+1)))
            except:
                pass


def tx():
    tx_dict = {'train':transforms.Compose([
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),

        'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        }
    return tx_dict


def get_dataset(world_size, rank, dataroot, phase, lim, transform, batch_size=64, shuffle=False, num_workers=8):


    dataset = SiameseDataset(dataroot, rank, phase, transform,lim=lim)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

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
                                            batch_size=batch_size)
    val_loader, val_sampler = get_dataset(world_size, rank, dataroot, 
                                        phase=phases[1], lim=50, 
                                        transform=tx_dict['val'], 
                                        batch_size=batch_size)

    # create model and optimizer
    encoder = Encoder(hidden_dim=256, num_encoder_layers=6, nheads=8)
    siamese_net = SiameseNetwork(encoder).to(rank)

    # Wrap the model with DistributedDataParallel
    siamese_net = DDP(siamese_net, device_ids=[rank], find_unused_parameters=False)

    optimizer = torch.optim.Adam(siamese_net.parameters(), lr=0.001)
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
        train_epoch(rank, siamese_net, optimizer, train_loader, epoch, num_epochs, running_loss=0)
        
        # Update the learning rate
        lr_scheduler.step()

        if rank==0:
            vloss, acc = validate(rank, siamese_net, val_loader)

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
            print('\nSaved weights to', save_path)

    # Clean up the process group
    cleanup()            


__all__ = ['pretrainer', 'train_epoch', 'SiameseDataset', 'Encoder', 'SiameseNetwork', 'contrastive_loss_cosine', 
'validate', 'tx', 'get_dataset', 'setup', 'cleanup']

if __name__ == '__main__':
    # devices = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device_ids = [0, 1]

    # root = '/media/jakep/eye/scr/dent/'
    # dataroot = '/media/jakep/eye/scr/pickle/'
    root = './'
    dataroot = '../pickle/'


    # ddp
    world_size = 2
    mp.spawn(pretrainer, args=(world_size, root, dataroot, ['train2', 'val2'], False), nprocs=world_size, join=True)

    # pretrain(devices, device_ids, root, dataroot)
