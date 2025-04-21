'''Import libraires'''
import os, yaml
from datetime import datetime
from easydict import EasyDict
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import KFold
from utils.setup_utils import (
    get_device,
    get_log_name,
)
from utils.training_utils import get_callbacks
import torch.nn as nn
from glob import glob
import scipy.io
import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from spikingjelly.clock_driven import neuron, functional, surrogate, layer



filelist = sorted(glob('D:/Project/AI/Datasets/BCI Competition V-3/Training set/*.mat'))

x_train = []
y_train = []
x_test = []
y_test = []

for idx, filename in enumerate(tqdm(filelist)):

    raw_train=scipy.io.loadmat(filename)
    raw_test=scipy.io.loadmat(filename.replace('Training', 'Validation'))

    x1=raw_train['epo_train'][0]['x'][0]
    y1=raw_train['epo_train'][0]['y'][0]

    x2=raw_test['epo_validation'][0]['x'][0]
    y2=raw_test['epo_validation'][0]['y'][0]

    x1=np.transpose(x1,(2,1,0))
    x1=x1.reshape((300,1,64,795))
    y1=np.argmax(y1,axis=0)

    x2=np.transpose(x2,(2,1,0))
    x2=x2.reshape((50,1,64,795))
    y2=np.argmax(y2,axis=0)

    if len(x_train) == 0:
        x_train = x1
        y_train = y1
        x_test = x2
        y_test = y2

    else:
        x_train = np.concatenate((x_train, x1), axis=0)
        y_train = np.concatenate((y_train, y1), axis=0)
        x_test = np.concatenate((x_test, x2), axis=0)
        y_test = np.concatenate((y_test, y2), axis=0)
        
class DeepSNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, time_steps, dropout_rate=0.4):
        super(DeepSNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), tau=2.0)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate.ATan(), tau=2.0)

        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.lif3 = neuron.LIFNode(surrogate_function=surrogate.ATan(), tau=2.0)
        self.time_steps = time_steps

    def forward(self, x):
        outputs = []
        for t in range(self.time_steps):
            x_t = x[:, t]
            x = self.lif1(self.dropout1(self.fc1(x_t)))
            x = self.lif2(self.dropout2(self.fc2(x)))
            x = self.lif3(self.dropout3(self.fc3(x)))
            outputs.append(x.unsqueeze(1))
        out = torch.cat(outputs, dim=1)
        return out

