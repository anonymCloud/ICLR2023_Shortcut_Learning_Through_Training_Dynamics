#!/usr/bin/env python
# coding: utf-8

'''
This script is for computing PD for a set of test images
given a model. Results are stored in a batch_info dictionary
which is finally saved as a pkl file 
'''

# --------------------- Class Definitions ---------------------

class center_crop(object):
    def crop_center(self, img):
        _, y, x = img.shape
        crop_size = np.min([y,x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[:, starty:starty + crop_size, startx:startx + crop_size]
    
    def __call__(self, img):
        return self.crop_center(img)

class normalize(object):
    def normalize_(self, img, maxval=255):
        img = (img)/(maxval)
        return img
    
    def __call__(self, img):
        return self.normalize_(img)

# --------------------- Function Definitions ---------------------

# input: densenet121 model, provide a hook function
# output: returns a model with hooks registered for all 58 layers
def register_hooks(model, hook):
    
    for idx,layer in enumerate(model.features.denseblock1):
        if idx%2==0:
            layer.register_forward_hook(hook)
        
    for idx,layer in enumerate(model.features.denseblock2):
        if idx%2==0:
            layer.register_forward_hook(hook)
        
    for idx,layer in enumerate(model.features.denseblock3):
        if idx%2==0:
            layer.register_forward_hook(hook)
        
    for idx,layer in enumerate(model.features.denseblock4):
        if idx%2==0:
            layer.register_forward_hook(hook)
        
    return model

def hook_feat_map(mod, inp, out):
    out = torch.nn.functional.interpolate(out,(8,8))
    feature_maps.append(torch.reshape(out, (out.shape[0],-1)))
  
def to_cpu(arr):
    for idx,x in enumerate(arr):
        arr[idx] = x.to('cpu')
    return arr

def print_memory_profile(s):
    # print GPU memory
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    print(s)
    print(t/1024**3,r/1024**3,a/1024**3)
    print('\n')

def compute_pred_depth(arr):
    last = arr[-1]

    if last==99 or arr[-2]==99:    # uncertain pd if last or penultimate layers are uncertain
        return -25  

    p_depth = 4
    for i in range(len(arr)-1):
        ele = arr[-1-(i+1)]
        if ele!=last:
            p_depth = (len(arr)-(i+1))*4 + 4
            break
    
    return p_depth

# ===================== Import Libraries =====================

from tqdm import tqdm   
import pandas as pd
import numpy as np
import os, sys, pdb
import torch, torchvision
import pickle
from PIL import Image
import argparse
import random

import models, datasets
from utils import *


# ===================== Parse Arguments =====================

parser = argparse.ArgumentParser()
parser.add_argument('ckpt_path', type=str, help='') 
parser.add_argument('pkl_path', type=str, help='') 
parser.add_argument('test_csv', type=str) 
parser.add_argument('save_path', type=str, help='path where final results will be saved') 
parser.add_argument('save_name', type=str, help='name of pickle file (eg: nih_shortcut)') 
parser.add_argument('--df_path_col', type=str, default='path', help='col name in the test_csv file having filenames of images') 
parser.add_argument('--cls_name', type=str, default='Pneumothorax', help='the labels for this correponding column in test_csv file is stored in saved pkl files for each image') 
parser.add_argument('--K', type=int, default=29)
parser.add_argument('--knn_pos_thresh', type=float, default=0.62, help='KNN considers N neighbours, and if the mean vote is larger than this thresh we take class as positive') 
parser.add_argument('--knn_neg_thresh', type=float, default=0.38, help='same as above, but if mean vote less than this thresh we treat point as negative class') 
parser.add_argument('--lp_norm', type=int, default=1, help='1 or 2. used to calculate KNN distances') 
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default='NIH') 
parser.add_argument('--img_size', type=int, default=128) 
parser.add_argument('--num_imgs', type=int, default=2000, help='number of test images from test_csv to consider') 
args = parser.parse_args()

# ===================== Seed =====================   
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

test_csv = args.test_csv
ckpt_path = args.ckpt_path
pkl_path = args.pkl_path
K = args.K
df_test = pd.read_csv(test_csv).sample(n=args.num_imgs,random_state=args.seed)

# ===================== Load Model =====================

feature_maps = []
model = torch.load(ckpt_path).to('cuda')
model = register_hooks(model, hook_feat_map)

# ===================== Dataset Transformations =====================

if args.dataset=='NIH':
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.img_size,args.img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(center_crop()),
        torchvision.transforms.Lambda(normalize())
    ])
elif args.dataset=='GithubCovid':
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.img_size,args.img_size)),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(center_crop()),
        torchvision.transforms.Lambda(normalize())
    ])
elif args.dataset=='HAM':
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),\
        torchvision.transforms.Resize((args.img_size,args.img_size)), 
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.ToTensor()
    ])
else:
    raise('Invalid dataset!')


# ===================== Storing Batch Statistics =====================

batch_info = {}
batch_info['readme'] = '---- K=%d ---- test_csv=%s ---- ckpt_path=%s ---- pkl_path=%s ----' %(K,test_csv,ckpt_path,pkl_path)
batch_info['paths'] = [] # paths of test images
batch_info['preds'] = [] # corresponding model predictions
batch_info['labels'] = [] # labels of the test images
batch_info['pd'] = [] # corresponding prediction depths
batch_info['layers_knn_mean'] = [] # for each test image we have a list of knn means for every layer
batch_info['layers_knn_mode'] = [] # for each test image we have a list of knn mode for every layer

print_memory_profile('Initial')

# ===================== Loop over test images and collect statistics =====================

for df_idx, img_path in enumerate(tqdm(df_test[args.df_path_col])):

    batch_info['paths'].append(img_path)
    with Image.open(img_path) as img:
        with torch.no_grad():
            img = transforms(img).unsqueeze(0).to('cuda')
            if img.shape[1]==4:
                img = img[:,0,:,:].unsqueeze(0)
            feature_maps = []
            out = model(img)
            print('Model output: ')
            print(torch.sigmoid(out))
            batch_info['preds'].append(round(float(torch.sigmoid(out)),2))
            batch_info['labels'].append(df_test.iloc[df_idx][args.cls_name])

            print_memory_profile('Model forward pass')

            # the below two lists are to store KNN nbr distances and labels across layers of dnet121
            # we need to loop over the val batches stored in pkl file and update the list across batches
            nbr_dist = [torch.empty((0)).to('cuda')]*len(feature_maps) # distance of neighbours
            nbr_labs = [torch.empty((0))]*len(feature_maps) # labels of neighbours

            with open(pkl_path, 'rb') as handle:
                # loop over val batches in pkl data
                for pkl_idx in tqdm(range(10000)):
                    info_dict = pickle.load(handle)
                    print_memory_profile('Pickle load')

                    # loop over layers in densenet
                    for layer_id,feat in tqdm(enumerate(feature_maps)):
                        X_i = feat.unsqueeze(1)  # (10000, 1, 784) test set
                        X_j = info_dict['feats'][layer_id].unsqueeze(0)  # (1, 60000, 784) train set
                        if args.lp_norm==2:
                            D_ij = ((X_i - X_j) ** 2).sum(-1)  # (10000, 60000) symbolic matrix of squared L2 distances
                        elif args.lp_norm==1:
                            D_ij = (abs(X_i - X_j)).sum(-1)  # (10000, 60000) symbolic matrix of squared L2 distances
                        else:
                            raise('Invalid lp_norm in arguments!')

                        ind_knn = torch.topk(-D_ij,K,dim=1)  # Samples <-> Dataset, (N_test, K)
                        lab_knn = info_dict['labels'][ind_knn[1]]  # (N_test, K) array of integers in [0,9]

                        # append knn preds for this layer (along with those in past batches)
                        nbr_dist[layer_id] = torch.cat((nbr_dist[layer_id],ind_knn[0]),dim=1)
                        nbr_labs[layer_id] = torch.cat((nbr_labs[layer_id],lab_knn.squeeze(2)),dim=1)

                    print_memory_profile('Pickle batch processed')
                    break_flag = (pkl_idx==info_dict['num_batches']-2) or (pkl_idx==info_dict['num_batches']-1)

                    # free GPU memory
                    del info_dict
                    torch.cuda.empty_cache()
                    print_memory_profile('After GPU memory freed')

                    if break_flag:
                        break # end of pickle objects                
                

    for test_id in range(len(nbr_labs[0])):    
        knn_preds_mode = []  # layer-wise final KNN classification preds         
        knn_preds_mean = []  # layer-wise final KNN classification preds  

        for layer_id in range(len(feature_maps)):
            topk_inds = torch.topk(nbr_dist[layer_id],K)  # Samples <-> Dataset, (N_test, K)
            topk_labs = nbr_labs[layer_id][test_id][topk_inds[1][test_id]].unsqueeze(0)
            knn_preds_mode.append(int(topk_labs.squeeze().mode()[0]))
            knn_preds_mean.append(round(float(topk_labs.mean(dim=1)),2))

        print('Test Image: %d' %(test_id))
        print(knn_preds_mode,knn_preds_mean)
        print('\n')
        batch_info['layers_knn_mean'].append(knn_preds_mean)
        batch_info['layers_knn_mode'].append(knn_preds_mode)
        if args.knn_pos_thresh==0.5 and args.knn_neg_thresh==0.5:
            batch_info['pd'].append(compute_pred_depth(knn_preds_mode))
        else:
            arr = knn_preds_mean
            for arr_idx,ele in enumerate(arr):
                if ele>args.knn_pos_thresh:
                    arr[arr_idx] = 1
                elif ele<args.knn_neg_thresh:
                    arr[arr_idx] = 0
                else:
                    arr[arr_idx] = 99   # its between pos thresh and neg thresh, means uncertain value
            print('Using pos and neg thresh we get: ')
            print(arr)
            batch_info['pd'].append(compute_pred_depth(arr))


# ===================== Save results =====================

with open(os.path.join(args.save_path,args.save_name+'.pkl'), 'wb') as handle:
    pickle.dump(batch_info, handle)