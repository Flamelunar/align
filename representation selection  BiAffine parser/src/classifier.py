import torch
import torch.nn as nn
from torch.autograd import Variable
# import numpy as np
# from torch.nn import functional, init
from common import *
from nn_modules import *
from flip_gradient import *
import torch.nn.functional as F


class ClassificationD(nn.Module):
    def __init__(self, name, input_size, hidden_size, activation=None):
        super(ClassificationD, self).__init__()
        self._name = name
        middle_size = input_size
        '''
        middle_size = int(input_size/2)
        self.linear = nn.Linear(in_features=input_size, out_features=middle_size)
        weights = orthonormal_initializer(middle_size, input_size)
        self.linear.weight.data = torch.from_numpy(weights)
        self.linear.weight.requires_grad = True
        b = np.zeros(middle_size, dtype=data_type)
        self.linear.bias.data = torch.from_numpy(b)
        self.linear.bias.requires_grad = True

        self._activate = (activation or (lambda x: x))
        assert(callable(self._activate))
        '''
        
        self.dropout=nn.Dropout(0.3)
        self.linear1 = nn.Linear(in_features=middle_size, out_features=hidden_size)
        weights1 = orthonormal_initializer(hidden_size, middle_size)
        self.linear1.weight.data = torch.from_numpy(weights1)
        self.linear1.weight.requires_grad = True
        #b1 = np.zeros(hidden_size, dtype=data_type)
        #self.linear1.bias.data = torch.from_numpy(b1)
        #self.linear1.bias.requires_grad = True


    @property
    def name(self):
        return self._name

    def forward(self, x):
        #x_grl = flip_gradient(x)
        #x_grl = GRLnew.apply(x, 1e-5)
        #y = self._activate(self.linear(x_grl))
        y = GRLnew.apply(x, 1e-5)
        #if is_training:
        y = self.dropout(y)
        #y = self._activate(self.linear(x_grl))
        y1 = self.linear1(y)
        return y1
    
    @staticmethod                                                                                                               
    def adversary_loss(score, domains, total_words):
        batch_size, len1, len2 = score.size()
        loss = F.cross_entropy(score.contiguous().view(batch_size * len1, len2),domains.view(batch_size * len1), ignore_index = 0)
        loss = loss / total_words 
        return loss
    
    @staticmethod
    def compute_accuray(score, true_labels):
        batch_size, len1, len2 = score.size()
        nscore= score.contiguous().view(batch_size * len1, len2)
        ntrue_labels= true_labels.view(batch_size * len1)
        total = ntrue_labels.size()[0]
        pred_labels = nscore.data.max(1)[1].cpu()
        pred_labels[ntrue_labels.eq(0)]=0
        #print("pred_labels",pred_labels)
        #print("true_labels",true_labels)
        correct = pred_labels.eq(ntrue_labels.cpu()).cpu().sum().item()
        ignore_word_nums = ntrue_labels.eq(0).sum().item()
        correct =correct - ignore_word_nums
        total = total - ignore_word_nums
        accuray = correct/total
        print("domain classificay accuray, correct, total", accuray, correct, total)


EPS = 1e-20
def avg_pooling(hidden, masks):
    sum_hidden = torch.sum(hidden, 1)
    len = torch.sum(masks, -1, keepdim=True)
    hidden = sum_hidden / (len + EPS)
    return hidden


class ClassificationDnew(nn.Module):
    def __init__(self, name, input_size, hidden_size, activation=None):
        super(ClassificationDnew, self).__init__()
        self._name = name
        self.dropout=nn.Dropout(0.3)
        '''
        middle_size= int(input_size/2)
        self.linear = nn.Linear(input_size, middle_size)
        weights = orthonormal_initializer(middle_size, input_size)
        self.linear.weight.data = torch.from_numpy(weights)
        self.linear.weight.requires_grad = True
        b = np.zeros(middle_size, dtype=data_type)
        self.linear.bias.data = torch.from_numpy(b)
        self.linear.bias.requires_grad = True

        self.MLP = MLPLayer('mlp_classifier', activation=nn.LeakyReLU(0.1), input_size = middle_size, hidden_size = middle_size)
        '''
        middle_size=input_size
        self.linear1 = nn.Linear(middle_size, hidden_size, False)
        weights1 = orthonormal_initializer(hidden_size, middle_size)
        self.linear1.weight.data = torch.from_numpy(weights1)
        self.linear1.weight.requires_grad = True
        #b1 = np.zeros(middle_size, dtype=data_type)
        #self.linear1.bias.data = torch.from_numpy(b1)
        #self.linear1.bias.requires_grad = True


    @staticmethod                                                                                                               
    def adversary_loss(score, domain_ids):
        domain_ids = domain_ids -1
        loss = F.cross_entropy(score, domain_ids,ignore_index=-1)
        return loss
    
    @staticmethod
    def compute_accuray(score, true_labels):
        total = true_labels.size()[0]
        pred_labels = score.data.max(1)[1].cpu()
        true_labels = true_labels -1
        #print("pred_labels",pred_labels)
        #print("true_labels",true_labels)
        correct = pred_labels.eq(true_labels.cpu()).cpu().sum().item()
        accuray = correct/total
        print("domain classificay accuray, correct, total", accuray, correct, total)

    @property
    def name(self):
        return self._name

    def forward(self, lstm_hidden, masks,is_training):
        x = avg_pooling(lstm_hidden, masks)
        #x = self.linear(x)
        #x_grl = flip_gradient(x)
        
        #x_grl = GRLnew.apply(x, 1e-5)
        #y = self.MLP(x_grl)
        y = GRLnew.apply(x, 1e-5)
        if is_training:
            y = self.dropout(y)
        score = self.linear1(y)
        return score


