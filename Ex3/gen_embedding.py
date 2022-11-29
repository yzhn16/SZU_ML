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
    
    def forward(self, x):
        emb = self._fc(self.backbone(x))
        prob = self.act(self.fc(emb))
        return torch.cat([emb, prob], -1)
    
def gen_onehot(x):
    res = [0] * 55
    for i in range(55):
        cur = x['type_{}'.format(i)]
        if cur != '':
            res[_dict[cur]] += 1
        else:
            break
    return res





class PreDataset(Dataset):
    def __init__(self, root, path, stage='train'):
        self.root = root
        
        names = ['file', 'age', 'gender'] + ['type_{}'.format(i) for i in range(55)]
        df = pd.read_csv(path, sep='\t', names=names, low_memory=False).fillna('')
        df['y'] = df.apply(lambda x: gen_onehot(x), axis=1)
        self.label = {row['file']: row['y'] for index, row in df.iterrows()}
        
        self.keys = df['file'].tolist()

            
    def __len__(self):
        
        return len(self.keys)
    
    
    def __getitem__(self, idx):
        x = np.array(pd.read_csv(os.path.join(self.root, self.keys[idx]), sep=' ').values).T
        y = np.array(self.label[self.keys[idx]])
        
        x1 = x.copy()
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
        
        return self.keys[idx], \
               torch.tensor(x.copy(), dtype=torch.float32), \
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
    


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    train_dataset = PreDataset('autodl-tmp/合肥高新-心电人机智能/train/', 'autodl-tmp/合肥高新-心电人机智能/hf_round1_label.txt')
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    test_dataset = PreDataset('autodl-tmp/合肥高新-心电人机智能/testA/', 'autodl-tmp/合肥高新-心电人机智能/hf_round1_subA.txt')
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)
    
    
    print('infer on', len(train_dataset), '+', len(test_dataset))
    
    model = PreModel()
    model.load_state_dict(torch.load('./model_SL.pth'))
    model.to(device)
    
    model.eval()
    
    keys = []
    embs = []
    for i, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader)):
        key, x, _, _, _ = data
            
        x = x.to(device)
        emb = model(x)
        
        keys.extend(list(key))
        embs.append(emb.detach().cpu().numpy())
        
    train_df = pd.DataFrame(np.vstack(embs), columns=['emb_{}'.format(i) for i in range(128)] + ['prob_{}'.format(i) for i in range(55)])
    train_df['key'] = keys
    train_df.to_csv('./train_embeddings.csv', index=False)
    
    keys = []
    embs = []
    for i, data in tqdm(enumerate(test_dataloader, 0), total=len(test_dataloader)):
        key, x, _, _, _ = data
            
        x = x.to(device)
        emb = model(x)
        
        keys.extend(list(key))
        embs.append(emb.detach().cpu().numpy())
    test_df = pd.DataFrame(np.vstack(embs), columns=['emb_{}'.format(i) for i in range(128)] + ['prob_{}'.format(i) for i in range(55)])
    test_df['key'] = keys
    test_df.to_csv('./test_embeddings.csv', index=False)