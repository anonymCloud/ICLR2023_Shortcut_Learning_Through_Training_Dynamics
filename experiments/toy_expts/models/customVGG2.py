'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import pickle
import numpy as np

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class customVGG2(nn.Module):
    def __init__(self, vgg_name, train_embs_path, layer_id, train_emb_idx, num_channels=3):
        super(customVGG2, self).__init__()
        self.num_channels = num_channels
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

        # required for custom forward pass:
        self.train_embs_path = train_embs_path
        self.layer_id = layer_id
        self.train_emb_idx = train_emb_idx
        self.test_feat_1d = None
        self.nbr_feats = None
        self.nbr_labels = None
        self.s = None

        # load train embeddings for soft-KNN
        with open(self.train_embs_path, 'rb') as handle:            
            self.info_dict = pickle.load(handle)

    def l1_norm(self, X_i, X_j):
        return (abs(X_i - X_j)).sum(-1)

    def l2_norm(self, X_i, X_j):
        return ((X_i - X_j) ** 2).sum(-1)

    def get_nbr_info(self, x):
        # get the nearest K neighbours for test img
        K = 29

        test_feat = self.features[:(self.layer_id+1)](x)
        self.test_feat_1d = torch.reshape(test_feat, (test_feat.shape[0],-1))
        
        info_dict = self.info_dict
        X_i = self.test_feat_1d.unsqueeze(1)  # (10000, 1, 784) test set
        X_j = info_dict['feats'][self.train_emb_idx].unsqueeze(0)  # (1, 60000, 784) train set
        D_ij = (abs(X_i - X_j)).sum(-1) 
        ind_knn = torch.topk(-D_ij,K,dim=1)  # Samples <-> Dataset, (N_test, K)
        lab_knn = info_dict['labels'][ind_knn[1]]  # (N_test, K) array of integers in [0,9]

        # free GPU memory
        del info_dict
        torch.cuda.empty_cache()

        # feed KNN nbr info of the test img to model
        nbr_inds = np.array(ind_knn[1][0].detach().cpu())
        self.nbr_feats = X_j[0,nbr_inds,:].to('cuda')
        self.nbr_labels = lab_knn.squeeze()
        self.s = float(torch.median(-ind_knn[0]).detach().cpu()) # for the median trick

    def forward(self, imgs):
        out = torch.empty((0,2)).to('cuda')
        for x in imgs:
            x = x.unsqueeze(0)
            self.get_nbr_info(x)
            x_feat_1d = self.test_feat_1d
            nr = 0
            dr = 0
            pred_cls = int(self.nbr_labels.mode()[0]) # predicted class based, argmax of the KNN
            for nbr in self.nbr_feats[self.nbr_labels==pred_cls]:    
                nr = nr + torch.exp(-self.l1_norm(x_feat_1d,nbr)/self.s)
            for nbr in self.nbr_feats:    
                dr = dr + torch.exp(-self.l1_norm(x_feat_1d,nbr)/self.s)
            frac = torch.cat((nr/dr,1-nr/dr)).unsqueeze(0)
            out = torch.cat((out,frac))

        return out  # probability value
    
    def simpleForward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.num_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)