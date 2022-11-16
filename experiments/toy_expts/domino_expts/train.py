#!/usr/bin/env python
# coding: utf-8

import torch, torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from PIL import Image
import random, os, sys, argparse
from pathlib import Path
from tqdm import tqdm
import pickle
from PIL import Image

sys.path.insert(0,'/xxx/home/xxx/xxx22p/xxx/projects/shortcut_detection_and_mitigation/experiments/toy_expts/')
from models import *

# Train Model

# ===================== Parse Arguments =====================

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet18') 
parser.add_argument('--seed', type=int, default=0) 
parser.add_argument('--train_csv', type=str, default='/xxx/home/xxx/xxx22p/xxx/projects/shortcut_detection_and_mitigation/experiments/toy_expts/domino_expts/top_mnist_bot_kmnist_2class_ro_1p0.csv') 
parser.add_argument('--test_csv', type=str, default='/xxx/home/xxx/xxx22p/xxx/projects/shortcut_detection_and_mitigation/csv_files/top_blank_bot_kmnist_2class_spcorr_1p0.csv') 
parser.add_argument('--expt_name', type=str, default='resnet18_top_mnist_bot_kmnist') 
parser.add_argument('--save_dir', type=str, default='/xxx/home/xxx/xxx22p/xxx/projects/shortcut_detection_and_mitigation/experiments/toy_expts/domino_expts/output/') 
args = parser.parse_args()

# user hyperparams
model = args.model 
train_csv = args.train_csv
test_csv = args.test_csv
expt_name = args.expt_name
save_dir = args.save_dir
seed = args.seed

num_epochs = 11
lr = 0.1
num_ch = 3 # num of channels in image
num_embs = 2000 # 1500 for mnist, 10k for cifar10
K = 29 # K neighbours
num_test_imgs = 20 # num of test images for plotting PD
lp_norm = 1 # for computing KNN
knn_pos_thresh = 0.5
knn_neg_thresh = 0.5

# Setting the seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv).sample(500)
# df_test = df_test[df_test['test_split']==0].sample(500)

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, class_field, transform, csv_path=None, df=None):
        self.class_field = class_field        
        self.transform = transform
        if df is not None:
            self.df = df
        else:
            self.csvpath = csvpath
            self.df = pd.read_csv(self.csvpath)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = str(self.df.iloc[idx]["path"])
        lab = int(self.df.iloc[idx][self.class_field])
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)  

        return {"img":img, "lab":lab, "idx":idx, "file_name" : img_path}

trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((64,32)),
])
trainset = ToyDataset(class_field=['bottom'], transform=trans, df=df_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
testset = ToyDataset(class_field=['bottom'], transform=trans, df=df_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True, num_workers=2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy


# Model
print('==> Building model..')
if model=='resnet18':
    net = ResNet18(num_channels=num_ch)
elif model=='vgg16':
    net = VGG('VGG16',num_channels=num_ch)
elif model=='densenet121':
    net = DenseNet121()
net.linear = nn.Linear(in_features=1024,out_features=10,bias=True)
net = net.to(device)
    
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, batch in enumerate(tqdm(trainloader)):
        inputs = batch['img']
        targets = batch['lab']
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                 % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(testloader)):
            inputs = batch['img']
            targets = batch['lab']
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                 % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
#     if acc > best_acc:
#         print('Saving..')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
# #         if not os.path.isdir(f'{args['expt_name']}_checkpoint'):
# #             os.mkdir('checkpoint')
#         torch.save(state, os.path.join(save_dir,f'{expt_name}_ep{epoch}.pt'))
#         best_acc = acc
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    torch.save(state, os.path.join(save_dir,f'{expt_name}.pt'))


for epoch in range(num_epochs):
    train(epoch)
    test(epoch)
    scheduler.step()