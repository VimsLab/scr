import os
import re
import torch
import pickle
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms as transforms

from PIL import Image
from glob import glob
from tqdm import tqdm
from pathlib import Path
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data.distributed import DistributedSampler

# os.environ["RANK"] = "0"
# os.environ["WORLD_SIZE"] = "2"
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12345'


# dist.init_process_group(backend="nccl")
# rank = dist.get_rank()
# world_size = dist.get_world_size()

# print(rank, world_size)
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format

class SiameseDataset(Dataset):
    def __init__(self, root, phase, transform=None):

        self.file_list = glob(os.path.join(root, phase + '/*.pkl'))
        self.transform = transform

        pos_file = phase+'positive.pkl'
        neg_file = phase+'negative.pkl'

        if not (Path(pos_file).is_file() and Path(neg_file).is_file()):
            print(f"\nCreating positve and negative pairs for {phase} since they don't exist yet")
            self.positive_pairs = []
            self.negative_pairs = []

            neg_limit = 1000

            # Create pairs
            print(('\n' + '%22s' * 3) % ('Image', 'Positive', 'Negative'))
            pbar = tqdm(enumerate(self.file_list), total=len(self.file_list), bar_format=TQDM_BAR_FORMAT)
            for i, fn_i in pbar:
                pid, tid = self._get_person_id(fn_i)
                pos = 0
                neg = 0

                for j, fn_j in enumerate(self.file_list[i+1:]):
                    pjd, tjd = self._get_person_id(fn_j)
                    
                    # if different person id, make negative examples
                    if pid != pjd:
                        self.negative_pairs.append((fn_i, fn_j))
                        neg+=1

                    # if same person id and same test id, make positive example
                    elif tid == tjd:
                        self.positive_pairs.append((fn_i, fn_j))
                        pos+=1

                    pbar.set_description(('%22s' * 3) % (f'{fn_i}', pos, neg))

                    if neg>neg_limit:
                        break

            # self.negative_pairs = random.sample(self.negative_pairs, min(len(self.negative_pairs), int(2*len(self.positive_pairs))))
            with open(pos_file, 'wb') as f:
                pickle.dump(self.positive_pairs, f)

            with open(neg_file, 'wb') as f:
                pickle.dump(self.negative_pairs, f)

        else:
            print(f"\nLoading positive and negative pairs from pickled lists of {phase}")
            with open(pos_file, 'rb') as f:
                self.positive_pairs = pickle.load(f)

            with open(neg_file, 'rb') as f:
                self.negative_pairs = pickle.load(f)

        print(f'{phase} dataset has {len(self.positive_pairs)} positive pairs and {len(self.negative_pairs)} Negative pairs.')
        print(f'Ratio of negative to positive samples = {len(self.negative_pairs)/len(self.positive_pairs)}')
        self.all_pairs = self.positive_pairs + self.negative_pairs
        random.shuffle(self.all_pairs)



    def _get_person_id(self, filename):
        # define the pattern using regular expressions eg: '104L20_5.pkl'
        # get the test id eg: '104L20_'
        # get person id eg: '104L'

        fn = filename.split('/')[-1]
        
        # testid = fn.split('_')[0]+'_'
        lr = fn.find('L')+1
        if not lr:
            lr = fn.find('R')+1

        pid = fn[:lr]
        testid = fn[:lr+1]

        return pid, testid



    def __getitem__(self, index):
        filename1, filename2 = self.all_pairs[index]

        with open(filename1, "rb") as f:
            data1 = pickle.load(f)

        with open(filename2, "rb") as f:
            data2 = pickle.load(f)

        images1 = Image.fromarray(data1["img"][0]).convert('RGB').resize((256, 224))
        images2 = Image.fromarray(data2["img"][0]).convert('RGB').resize((256, 244))

        # Augmentations
        if self.transform:
            images1 = self.transform(images1)
            images2 = self.transform(images2)

        # Set target based on whether the pair is positive or negative
        target = torch.tensor(1 if (filename1, filename2) in self.positive_pairs else 0)

        return images1, images2, filename1, filename2, target


    def __len__(self):
        return len(self.all_pairs)


class DETREncoder(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=3, nhead=8):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead),
            num_layers=num_layers
        )
        self.fc = nn.Linear(in_features=2048, out_features=hidden_dim)

    def forward(self, x):
        features = self.backbone(x)  # (batch_size, 2048, H/32, W/32)
        features = features.mean(dim=[2, 3])  # (batch_size, 2048)
        features = self.fc(features)  # (batch_size, hidden_dim)
        features = features.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        encoded = self.transformer_encoder(features)  # (batch_size, 1, hidden_dim)
        encoded = encoded.squeeze(1)  # (batch_size, hidden_dim)
        return encoded


class SiameseNetwork(nn.Module):
    def __init__(self, encoder):
        super(SiameseNetwork, self).__init__()
        self.encoder = encoder
        
    def forward(self, x1, x2):
        # with torch.no_grad():
        embedding1 = self.encoder(x1)
        embedding2 = self.encoder(x2)
        return embedding1, embedding2
    
def get_loss(embeddings1, embeddings2, targets, margin=2.0):
        distances = F.pairwise_distance(embeddings1, embeddings2)
        loss = torch.mean((1-targets) * torch.pow(distances, 2) +
                          targets * torch.pow(torch.clamp(margin - distances, min=0.0), 2))
        return loss


def validate(siamese_net, val_loader, device, thres=0.55):
    siamese_net.eval()
    with torch.no_grad():
        total_loss = 0
        correct = 0
        total = 0

        print(('\n' + '%22s' * 4) % ('Device', 'Correct', 'Accuracy', 'Loss'))
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        for batch_idx, (x1, x2, f1, f2, targets) in pbar:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            embeddings1, embeddings2 = siamese_net(x1, x2)
            loss = get_loss(embeddings1, embeddings2, targets)

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

            pbar.set_description(('%22s'*2 +'%22.4g' * 2) % (f'{pred.device}', f'{correct}/{total}', correct/total, total_loss/(batch_idx+1)))


        # calculate average loss and accuracy
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

    return avg_loss, accuracy


def train_epoch(siamese_net, train_loader, epoch, epochs, device, running_loss=0):  
    siamese_net.train()
    print(('\n' + '%22s' * 4) % ('Device', 'Epoch', 'GPU Mem', 'Loss'))

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:+10b}')
    for batch_idx, (x1, x2, f1, f2, targets) in pbar:
        # print(f1, f2, targets)
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        embeddings1, embeddings2 = siamese_net(x1, x2)
        loss = get_loss(embeddings1, embeddings2, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # print statistics

        running_loss = running_loss+loss.item()
        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
        pbar.set_description(('%22s'*3 + '%22.4g' * 1) % (f'{torch.cuda.current_device()}', f'{epoch}/{epochs - 1}', mem, running_loss/(batch_idx+1)))



if __name__ == "__main__":
    # example usage
    devices = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device_ids = [0, 1]
    num_epochs = 134

    root = '/media/jakep/eye/scr/dent/'
    dataroot = '/media/jakep/eye/scr/pickle/'
    batch_size = 64
    train_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SiameseDataset(dataroot, phase='train2', transform=train_transforms)
    # train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = SiameseDataset(dataroot, phase='val2', transform=val_transforms)
    # val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)


    # create model and optimizer
    encoder = DETREncoder(hidden_dim=256, num_layers=6, nhead=8)
    siamese_net = SiameseNetwork(encoder).to(devices)
    siamese_net = DataParallel(siamese_net, device_ids=device_ids)
    optimizer = torch.optim.Adam(siamese_net.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_accuracy = 0
    # Train the network
    for epoch in range(num_epochs):
        # train_sampler.set_epoch(epoch)
        train_epoch(siamese_net, train_loader, epoch, num_epochs, torch.cuda.current_device(), running_loss=0)
        
        # Update the learning rate
        lr_scheduler.step()
        vloss, acc = validate(siamese_net, val_loader, torch.cuda.current_device())

        if acc>=best_accuracy:
            best_accuracy = acc
            save_path = root + 'best.pth'
        else:
            save_path = root + 'last.pth'

        # if rank==0:
        checkpoint = {
                'epoch': epoch,
                'model_state_dict': siamese_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_accuracy,
            }
        torch.save(checkpoint, save_path)
        

