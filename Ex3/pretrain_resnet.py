import os
import random
import math

import numpy as np
import pandas as pd
from scipy import signal

from tqdm import tqdm

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.utils.model_zoo as model_zoo
from sklearn.metrics import f1_score, roc_auc_score

arr = pd.read_csv('autodl-tmp/合肥高新-心电人机智能/hf_round1_arrythmia.txt', encoding='GBK',sep='\t', header=None)[0]
_dict = {k: v for v, k in enumerate(arr)}
        
        
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=False)
    
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, bias=False, padding=3)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=11, stride=stride,
                               padding=5, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=7, bias=False, padding=3)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(8, 64, kernel_size=15, stride=2, padding=7,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x
    
class PreModel(nn.Module):
    def __init__(self):
        super(PreModel, self).__init__()
        block = BasicBlock # Bottleneck [3, 4, 6, 3]
        self.backbone = ResNet(block, [2, 2, 2, 2]) #ResNet(BasicBlock, [2, 2, 2, 2])
        self._fc = nn.Linear(512 * block.expansion, 128)
        
        self.fc = nn.Linear(128, 55)
        self.fc1 = nn.Linear(128, 2)
        self.act = nn.Sigmoid()
    
    def forward(self, x, x1):
        x = self._fc(self.backbone(x))
        
        return self.act(self.fc(x)), self.act(self.fc1(x))

    
def gen_onehot(x):
    res = [0] * 55
    for i in range(55):
        cur = x['type_{}'.format(i)]
        if cur != '':
            res[_dict[cur]] = 1
        else:
            break
    return res
  


class PreDataset(Dataset):
    def __init__(self, root, path, keys):
        self.root = root
        
        names = ['file', 'age', 'gender'] + ['type_{}'.format(i) for i in range(55)]
        df = pd.read_csv(path, sep='\t', names=names, low_memory=False).fillna('')
        df['y'] = df.apply(lambda x: gen_onehot(x), axis=1)
        self.label = {row['file']: row['y'] for index, row in df.iterrows()}
        
        self.keys = keys#df['file'].tolist()

            
    def __len__(self):
        
        return len(self.keys)
    
    
    def __getitem__(self, idx):
        x = np.array(pd.read_csv(os.path.join(self.root, self.keys[idx]), sep=' ').values).T
        y = np.array(self.label[self.keys[idx]])
        
        length = 1024
        split = random.choice(range(5001 - length)) # 4999 - 512 + 1
        x1 = x.copy()[:, split: split + length]# + np.random.randn(length) * np.sqrt(50)
        y1 = np.array([0, 0])
        choice = random.choice([1, 2, 3, 4])
        if choice == 2:
            x1 = x1 * (-1)
            y1 = np.array([1, 0])
        elif choice == 3:
            x1 = np.fliplr(x1)
            y1 = np.array([0, 1])
        elif choice == 4:
            x1 = np.fliplr(x1 * (-1))
            y1 = np.array([1, 1])
            
        
        
        return torch.tensor(x.copy(), dtype=torch.float32), \
               torch.tensor(y.copy(), dtype=torch.float32), \
               torch.tensor(x1.copy(), dtype=torch.float32), \
               torch.tensor(y1.copy(), dtype=torch.float32)
        

class WeightedMultilabel(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        return (loss * self.weights).mean()
    

def f1(y_true, y_pred, threshold=0):
    return f1_score(y_true.flatten(), np.where(y_pred > threshold, 1, 0).flatten())

# cnt
names = ['key', 'age', 'gender'] + ['arr_{}'.format(i) for i in range(55)]
train_info = pd.read_csv('autodl-tmp/合肥高新-心电人机智能/hf_round1_label.txt', sep='\t', names=names, low_memory=False).fillna(-1)
for i in range(55):
    train_info['type_{}'.format(i)] = 0
def gen_onehot_(x):
    for i in range(55):
        cur = x['arr_{}'.format(i)]
        if cur != -1:
            x['type_{}'.format(_dict[cur])] = 1
        else:
            break
    return x
train_info = train_info.apply(lambda x: gen_onehot_(x), axis=1)
cnt = [1 / np.log(sum(train_info['type_{}'.format(i)])) for i in range(55)]

    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_keys = pd.read_csv('autodl-tmp/合肥高新-心电人机智能/hf_round1_label.txt',
                             sep='\t', low_memory=False, usecols=[0], header=None)[0]
    
    valid_keys = train_keys.sample(frac=0.2, random_state=3407).tolist()
    train_keys = [k for k in train_keys.tolist() if k not in valid_keys]
    test_keys = pd.read_csv('autodl-tmp/合肥高新-心电人机智能/heifei_round1_ansA_20191008.txt',
                             sep='\t', low_memory=False, usecols=[0], header=None)[0]
    
    train_dataset = PreDataset('autodl-tmp/合肥高新-心电人机智能/train/',
                               'autodl-tmp/合肥高新-心电人机智能/hf_round1_label.txt', train_keys)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    valid_dataset = PreDataset('autodl-tmp/合肥高新-心电人机智能/train/',
                               'autodl-tmp/合肥高新-心电人机智能/hf_round1_label.txt', valid_keys)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True, num_workers=8)
    test_dataset = PreDataset('autodl-tmp/合肥高新-心电人机智能/testA/',
                              'autodl-tmp/合肥高新-心电人机智能/heifei_round1_ansA_20191008.txt', test_keys)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)
    
    print('train on', len(train_dataset), 'valid on', len(valid_dataset))
    
    model = PreModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    w = torch.tensor(cnt, dtype=torch.float).to(device)
    criterion = nn.BCELoss() # WeightedMultilabel(w)
    criterion1 = nn.MSELoss()
    
    max_auc = 0
    flag = 0
    for epoch in range(50):
        model.train()
        
        y_trues = []
        y_preds = []
        _hit = 0
        _total = 0
        pbar = tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader))
        for i, data in pbar:
            x, y, x1, y1 = data
            
            x = x.to(device)
            y = y.to(device)
            x1 = x1.to(device)
            y1 = y1.to(device)
            
            optimizer.zero_grad()
            
            yhat, y1hat = model(x, x1)
            loss = criterion(yhat, y) + 1e-8 * criterion1(y1hat, y1)
            
            loss.backward()
            optimizer.step()
            # print(y, yhat)
            y_numpy = y.detach().cpu().numpy()
            yhat_numpy = yhat.detach().cpu().numpy()
            y_trues.append(y_numpy)
            y_preds.append(yhat_numpy)
            
            _hit += int(((y1 == (y1hat > 0.5)).sum(axis=1) == 2).sum())
            _total += len(y1)
            
            pbar.set_postfix({'microAUC' : '{0:1.5f}'.format(roc_auc_score(y_numpy, yhat_numpy,average='micro'))})
            
        print('train macroAUC', roc_auc_score(np.vstack(y_trues), np.vstack(y_preds), average='macro'),
              'f1', f1(np.vstack(y_trues), np.vstack(y_preds), 0.5),
              'task1 acc', _hit / _total)
        
        
        model.eval()
        
        
        y_trues = []
        y_preds = []
        _hit = 0
        _total = 0
        for i, data in tqdm(enumerate(valid_dataloader, 0), total=len(valid_dataloader)):
            x, y, x1, y1 = data
            
            x = x.to(device)
            y = y.to(device)
            x1 = x1.to(device)
            y1 = y1.to(device)
            
            yhat, y1hat = model(x, x1)
            
            y_numpy = y.detach().cpu().numpy()
            yhat_numpy = yhat.detach().cpu().numpy()
            y_trues.append(y_numpy)
            y_preds.append(yhat_numpy)
            
            _hit += int(((y1 == (y1hat > 0.5)).sum(axis=1) == 2).sum())
            _total += len(y1)
            
        print('valid macroAUC', roc_auc_score(np.vstack(y_trues), np.vstack(y_preds), average='macro'), 
              'f1', f1(np.vstack(y_trues), np.vstack(y_preds), 0.5), 
              'task1 acc', _hit / _total)
        
        if(roc_auc_score(np.vstack(y_trues), np.vstack(y_preds), average='macro') > max_auc):
            max_auc = roc_auc_score(np.vstack(y_trues), np.vstack(y_preds), average='macro')
            print('saved')
            torch.save(model.state_dict(), './model_SSL.pth')
            flag = 0
        else:
            flag += 1
        
        if flag == 3:
            print('early stop')
            break
        
            
        y_trues = []
        y_preds = []
        _hit = 0
        _total = 0
        for i, data in tqdm(enumerate(test_dataloader, 0), total=len(test_dataloader)):
            x, y, x1, y1 = data
            
            x = x.to(device)
            y = y.to(device)
            x1 = x1.to(device)
            y1 = y1.to(device)
            
            yhat, y1hat = model(x, x1)
            
            y_numpy = y.detach().cpu().numpy()
            yhat_numpy = yhat.detach().cpu().numpy()
            y_trues.append(y_numpy)
            y_preds.append(yhat_numpy)
            
            _hit += int(((y1 == (y1hat > 0.5)).sum(axis=1) == 2).sum())
            _total += len(y1)
            
        print('test macroAUC', roc_auc_score(np.vstack(y_trues), np.vstack(y_preds), average='macro'),
              'f1', f1(np.vstack(y_trues), np.vstack(y_preds), 0.5),
              'task1 acc', _hit / _total)