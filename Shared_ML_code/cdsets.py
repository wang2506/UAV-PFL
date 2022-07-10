# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 13:05:56 2022

@author: ch5b2
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


import torch
import pickle
from torch import nn,autograd
from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
from copy import deepcopy

class RML(Dataset):
    def __init__(self,ldir,train=True):
        db_vec = [0,2,4,6,8,10,12,14,16,18] #
        train_x,train_y = [],[]
        test_x,test_y = [],[]
        for ind_db,db_vals in enumerate(db_vec):
            in_x = np.load(ldir+'/x_'+str(db_vals)+'.npy')
            in_x /= np.amax(in_x)
            in_x = in_x[:,np.newaxis,:,:]
            in_y = np.load(ldir+'/y_'+str(db_vals)+'.npy')
            
            ## segment dataset so that only first 4 labels are kept
            train = {i: [] for i in range(10)}
            # for index, label in enumerate(in_x):
            #     train[label].append(index)
            
            test = {i: [] for i in range(10)} 
            for index, label in enumerate(in_y):
                test[label].append(index)    
                train[label].append(index)

            in_x = np.concatenate([in_x[train[0]],in_x[train[1]],in_x[train[2]],in_x[train[3]]])
            in_y = np.concatenate([in_y[train[0]],in_y[train[1]],in_y[train[2]],in_y[train[3]]])
            
            x_train, x_test, y_train, y_test = \
                train_test_split(in_x, in_y, test_size=0.20, random_state=(25), shuffle=True)
            
            train_x.append(x_train)
            train_y.append(y_train)
            
            test_x.append(x_test)
            test_y.append(y_test)
        
        ovr_trainx = np.concatenate(train_x)
        ovr_trainy = np.concatenate(train_y)
        
        ovr_testx = np.concatenate(test_x)
        ovr_testy = np.concatenate(test_y)
        
        ### This works for both MLP and CNN
        # in_x = np.load(ldir+'/x_18.npy')
        # in_y = np.load(ldir+'/y_18.npy')
        # # in_x = np.reshape(in_x,(in_x.shape[0],in_x.shape[1]*in_x.shape[2])) #MLP
        
        # ## normalize
        # in_x /= np.amax(in_x)
        
        # in_x = in_x[:,np.newaxis,:,:] #CNN
        # # # in_x = np.reshape(in_x,(in_x.shape[0],in_x.shape[1],\
        # # #         int(np.sqrt(in_x.shape[2]*in_x.shape[3])),\
        # # #             int(np.sqrt(in_x.shape[2]*in_x.shape[3]))))
        # x_train, x_test, y_train, y_test = \
        #     train_test_split(in_x, in_y, test_size=0.20, random_state=(25), shuffle=True)
        
        if train == True:
            # self.x_data,self.y_data = x_train,y_train
            self.x_data,self.y_data = ovr_trainx,ovr_trainy
        else:
            # self.x_data,self.y_data = x_test,y_test
            self.x_data,self.y_data = ovr_testx,ovr_testy
        
    def __getitem__(self, i):
        return self.x_data[i], self.y_data[i]


# class MLP2(nn.Module):
#     def __init__(self,dim_in,dim_hidden,dim_out):
#         super(MLP2,self).__init__()
#         self.layer_input = nn.Linear(dim_in,dim_hidden)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout()
#         self.layer_hidden = nn.Linear(dim_hidden,dim_out)
#         self.softmax = nn.Softmax(dim=1)
#         # self.sigmoid = nn.Sigmoid()
        
#     def forward(self,x):
#         x = self.layer_input(x)
#         x = self.dropout(x)
#         x = self.relu(x)
#         x = self.layer_hidden(x)
#         # return self.sigmoid(x)
#         return self.softmax(x)

class CNNR(nn.Module):
    def __init__(self,nchannels,nclasses):
        super(CNNR, self).__init__()
        self.conv1 = nn.Conv2d(nchannels, 16, kernel_size=(2,5))#2, 1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1,4)) #2, 1)
        self.fc1 = nn.Linear(3872, 128)
        self.fc2 = nn.Linear(128, nclasses)

        # self.conv1 = nn.Conv2d(nchannels,256,kernel_size=(2,5))
        # self.conv2 = nn.Conv2d(256,128,kernel_size=(1,4))
        # self.conv3 = nn.Conv2d(128,64,kernel_size=(1,3))
        # self.conv4 = nn.Conv2d(64,64,kernel_size=(1,3))
        # self.drop = nn.Dropout(0.2)
        
        # self.fc1 = nn.Linear(7488, 128)
        # self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        # x = nn.MaxPool2d(2, 1)(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        # x = nn.MaxPool2d(2, 1)(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        # x = nn.ReLU()(self.conv1(x))
        # # x = self.drop(x)
        # x = nn.ReLU()(self.conv2(x))
        # # x = self.drop(x)
        # x = nn.ReLU()(self.conv3(x))
        # # x = self.drop(x)        
        # x = nn.ReLU()(self.conv4(x))
        # # x = self.drop(x)
        # x = torch.flatten(x,1)
        # x = nn.ReLU()(self.fc1(x))
        # x = self.fc2(x)
        # output = F.softmax(x,dim=1)
        return output

class segmentdataset(Dataset):
    def __init__(self,dataset,indexes):
        self.dataset = dataset
        self.indexes = indexes
    def __len__(self):
        return len(self.indexes)
    def __getitem__(self,item):
        image,label = self.dataset[self.indexes[item]]
        return image,label
    
class LocalUpdate(object):
    def __init__(self,device,bs,lr,epochs,dataset=None,indexes=None):
        self.device = device
        self.bs = bs
        self.lr = lr
        self.dataset = dataset
        self.indexes = indexes
        self.epochs = epochs
        self.ldr_train = DataLoader(segmentdataset(dataset,indexes),batch_size=bs,shuffle=True)
        # self.ldr_train = DataLoader(segmentdataset(dataset,indexes),batch_size=len(indexes),shuffle=True)
        self.loss_func = nn.CrossEntropyLoss()
        
    def train(self,net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(),lr=self.lr)#, momentum=0.5,weight_decay=1e-4) #l2 penalty
        epoch_loss = []
        for epoch in range(self.epochs):
            batch_loss = []
            
            for batch_indx,(images,labels) in enumerate(self.ldr_train):
                images,labels = images.to(self.device),labels.to(self.device)
                labels = labels.type(torch.long)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs,labels)
                loss.backward() #this computes the gradient
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net,net.state_dict(),(sum(batch_loss)/len(batch_loss))

class SGD_PFL(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v_{t+1} = \mu * v_{t} + g_{t+1} \\
                  p_{t+1} = p_{t} - lr * v_{t+1}

        where p, g, v and :math:`\mu` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
              v_{t+1} = \mu * v_{t} + lr * g_{t+1} \\
              p_{t+1} = p_{t} - v_{t+1}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                  weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_PFL, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_PFL, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            ## updates the parameters
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                
                
                # this one sort of works
                # d_p = torch.where(torch.isnan(d_p), torch.zeros_like(d_p), d_p)
                p.data.add_(-group['lr'], d_p)

        return loss



class LocalUpdate_trad_FO(object): #MLP 1e-3; CNN 1e-2
    def __init__(self,device,bs,lr1,lr2,epochs,dataset=None,indexes=None):
        self.device = device
        self.bs = bs
        self.lr1 = lr1
        self.lr2 = lr2
        self.dataset = dataset
        self.indexes = indexes
        self.epochs = epochs
        ###
        # prev ldr_train with bs/3, rather than all data
        ###
        # self.ind_split = int(len(indexes)/3)
        # self.ind1 = random.sample(indexes,self.ind_split)
        # self.ind2 = random.sample(indexes,self.ind_split)
        # self.ldr_train = DataLoader(segmentdataset(dataset,self.ind1),\
        #             batch_size=self.ind_split,shuffle=True)
        # self.ldr_train2 = DataLoader(segmentdataset(dataset,self.ind2),\
        #             batch_size=self.ind_split,shuffle=True)
        
        self.ldr_train = DataLoader(segmentdataset(dataset,indexes),\
                    batch_size=int(self.bs/3),shuffle=True)
        self.ldr_train2 = DataLoader(segmentdataset(dataset,indexes),\
                    batch_size=int(self.bs/3),shuffle=True)
        
            
        self.loss_func = nn.CrossEntropyLoss() #works for MLP
        # self.loss_func = nn.NLLLoss() #still fails for CNN
        
    def train(self,net):
        net.train()
        optimizer = SGD_PFL(net.parameters(),lr=self.lr1)#, momentum=0.5,weight_decay=1e-4)
        # optimizer2 = torch.optim.SGD(net.parameters(),lr=self.lr2, momentum=0.5,weight_decay=1e-4)
        
        epoch_loss = []
        for epoch in range(self.epochs):
            batch_loss = []
            
            # calculate the meta-function of SGD
            temp = deepcopy(net.state_dict())
            # print('start of LocalUpdate_HF_PFL')
            # print(temp['fc2.bias'])
            
            temp_params = [] #temp_params = deepcopy(net.parameters())
            for i,j in enumerate(net.parameters()):
                temp_params.append(deepcopy(j))
            temp_params_dict = deepcopy(net.state_dict())
            
            ## inner params obtain - step size - eta_1
            for batch_indx,(images,labels) in enumerate(self.ldr_train):
                images,labels = images.to(self.device),labels.to(self.device)
                labels = labels.type(torch.long)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs,labels)
                batch_loss.append(loss.item()) # init batch loss
                loss.backward()
                optimizer.step()
            
            # this produces the intermediate parameters - needed inner for all three terms
            temp_w_inner = deepcopy(net.state_dict()) #used to find intermediate loss            
            temp_w_inner_params = []
            for i,j in enumerate(net.parameters()):
                temp_w_inner_params.append(deepcopy(j))
            # print('w inner result')
            # print(temp_w_inner['fc2.bias'])
            
            ## need three gradients
            # grad 1: D_outer
            for batch_indx,(images,labels) in enumerate(self.ldr_train2):
                lr2_result = []
                
                images,labels = images.to(self.device),labels.to(self.device)
                labels = labels.type(torch.long)
                net.zero_grad()#, net_pos.zero_grad(), net_neg.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs,labels)           
                loss.backward()
                
                # manual grad calc here
                temp_inner_params = [tval for tval in net.parameters()] #deepcopy(net.parameters())
                for p1,p2 in enumerate(temp_inner_params): #the initial starting params
                    # this is w_i(t) = w_i(t-1) - lr2 * grad
                    lr2_result.append(temp_params[p1]-self.lr2 * p2.grad)  
                    # this becomes new temp_params
                    
                # load in new params, and continue mini batch process
                # update temp_params (which are the original parameters)
                p_count = 0
                for p_key in temp_params_dict.keys():
                    temp_params_dict[p_key] = lr2_result[p_count]
                    p_count += 1
                net.load_state_dict(temp_params_dict)
                temp_params = [val for val in net.parameters()]
                

            # print('everything put together params')
            # print(net.state_dict()['fc2.bias'])
            
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net,net.state_dict(),(sum(batch_loss)/len(batch_loss))

def test_img2(net_g, datatest,bs,indexes,device=torch.device('cpu')):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(segmentdataset(datatest,indexes),batch_size=bs,shuffle=True)
    
    for idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        
        ## added this for radioML dataset
        target = target.type(torch.long)
        
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100*correct.item() / len(data_loader.dataset)
    return accuracy, test_loss

if __name__ == '__main__':
    pwd = os.pardir
    cwd = os.getcwd()
    dtrain = RML(ldir=pwd+'/data/radio_ml/',train=True)
    dtest = RML(ldir=pwd+'/data/radio_ml/',train=False)
    
    # torch.manual_seed(1)
    # np.random.seed(1)
    # # determine the number of datapoints per label
    # train = {i: [] for i in range(10)}
    # for index, label in enumerate(dtrain.y_data):
    #     train[label].append(index)
    
    # test = {i: [0] for i in range(10)} 
    # for index, label in enumerate(dtest.y_data):
    #     test[label].append(index)
    
    # import matplotlib.pyplot as plt
    # for i in range(10):
    #     plt.figure()
    #     plt.title('plot label '+str(i))
    #     for j in range(1):
    #         temp = dtrain[train[i][j]][0] # for concat of input data
    #         temp = np.reshape(temp,(temp.shape[1]*temp.shape[2],temp.shape[0]))
    #         plt.plot(temp)
    #         # plt.plot(dtrain[train[i][j]][0])

    device = torch.device('cuda')
    d_in = 2*128
    # d_in = 128
    d_h = 64
    d_out = 10
    # global_net = MLP2(d_in,d_h,d_out).to(device)
    global_net = CNNR(1,4).to(device)
    # lr = 1e-1 #works better for MLP

    for i in range(20): #range(100)
        # if i < 10:
        #     lr = 5e-2
        # elif i < 20:
        #     lr = 1e-2
        # elif i < 30:
        #     lr = 5e-3
        # elif i < 40:
        #     lr = 1e-3
        # elif i < 50:
        #     lr = 5e-4
        # else:
        #     lr = 5e-4
        
        lr = 1e-2
        lr2 = 5e-2
        
        t_obj = LocalUpdate(device,bs=12,lr=lr,epochs=1,\
                    dataset=dtrain,indexes=range(dtrain.y_data.shape[0]))
        # t_obj = LocalUpdate_trad_FO(device,bs=12,lr1=lr,lr2=lr,epochs=1,\
        #             dataset=dtrain,indexes=range(dtrain.y_data.shape[0]))
        _,w,loss = t_obj.train(net=global_net)
        # print(w['layer_hidden.bias'])
        global_net.load_state_dict(w)
        
        c_acc, c_loss = \
            test_img2(global_net,dtest,\
            bs=12,indexes=range(dtest.y_data.shape[0]),device=device)
    
        print(c_acc,c_loss)


    ## save the parameters as a starting point
    pwd = os.path.dirname(cwd)
    with open(pwd+'/data/CNN_mlradio_w','wb') as f:
        pickle.dump(w,f)

