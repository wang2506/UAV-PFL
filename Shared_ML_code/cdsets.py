# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 13:05:56 2022

@author: ch5b2
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


# import torch
# import pickle
# from torch import nn,autograd
# from torch.utils.data import DataLoader,Dataset
# import numpy as np
# import torch.nn.functional as F

class RML(Dataset):
    def __init__(self,ldir,train=True):
        db_vec = [10,12,14,16,18] #0,2,4,6,8,
        train_x,train_y = [],[]
        test_x,test_y = [],[]
        for ind_db,db_vals in enumerate(db_vec):
            in_x = np.load(ldir+'/x_'+str(db_vals)+'.npy')
            in_x /= np.amax(in_x)
            in_x = in_x[:,np.newaxis,:,:]
            in_y = np.load(ldir+'/y_'+str(db_vals)+'.npy')
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

# class CNNR(nn.Module):
#     def __init__(self,nchannels,nclasses):
#         super(CNNR, self).__init__()
#         self.conv1 = nn.Conv2d(nchannels, 16, kernel_size=(2,5))#2, 1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=(1,4)) #2, 1)
#         self.fc1 = nn.Linear(3872, 128)
#         self.fc2 = nn.Linear(128, nclasses)

#         # self.conv1 = nn.Conv2d(nchannels,256,kernel_size=(2,5))
#         # self.conv2 = nn.Conv2d(256,128,kernel_size=(1,4))
#         # self.conv3 = nn.Conv2d(128,64,kernel_size=(1,3))
#         # self.conv4 = nn.Conv2d(64,64,kernel_size=(1,3))
#         # self.drop = nn.Dropout(0.2)
        
#         # self.fc1 = nn.Linear(7488, 128)
#         # self.fc2 = nn.Linear(128, 10)
        
#     def forward(self, x):
#         x = self.conv1(x)
#         x = nn.ReLU()(x)
#         # x = nn.MaxPool2d(2, 1)(x)
#         x = self.conv2(x)
#         x = nn.ReLU()(x)
#         # x = nn.MaxPool2d(2, 1)(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = nn.ReLU()(x)
#         x = self.fc2(x)
#         output = F.softmax(x, dim=1)
#         # x = nn.ReLU()(self.conv1(x))
#         # # x = self.drop(x)
#         # x = nn.ReLU()(self.conv2(x))
#         # # x = self.drop(x)
#         # x = nn.ReLU()(self.conv3(x))
#         # # x = self.drop(x)        
#         # x = nn.ReLU()(self.conv4(x))
#         # # x = self.drop(x)
#         # x = torch.flatten(x,1)
#         # x = nn.ReLU()(self.fc1(x))
#         # x = self.fc2(x)
#         # output = F.softmax(x,dim=1)
#         return output

# class segmentdataset(Dataset):
#     def __init__(self,dataset,indexes):
#         self.dataset = dataset
#         self.indexes = indexes
#     def __len__(self):
#         return len(self.indexes)
#     def __getitem__(self,item):
#         image,label = self.dataset[self.indexes[item]]
#         return image,label
    
# class LocalUpdate(object):
#     def __init__(self,device,bs,lr,epochs,dataset=None,indexes=None):
#         self.device = device
#         self.bs = bs
#         self.lr = lr
#         self.dataset = dataset
#         self.indexes = indexes
#         self.epochs = epochs
#         self.ldr_train = DataLoader(segmentdataset(dataset,indexes),batch_size=bs,shuffle=True)
#         # self.ldr_train = DataLoader(segmentdataset(dataset,indexes),batch_size=len(indexes),shuffle=True)
#         self.loss_func = nn.CrossEntropyLoss()
        
#     def train(self,net):
#         net.train()
#         optimizer = torch.optim.SGD(net.parameters(),lr=self.lr)#, momentum=0.5,weight_decay=1e-4) #l2 penalty
#         epoch_loss = []
#         for epoch in range(self.epochs):
#             batch_loss = []
            
#             for batch_indx,(images,labels) in enumerate(self.ldr_train):
#                 images,labels = images.to(self.device),labels.to(self.device)
#                 labels = labels.type(torch.long)
#                 net.zero_grad()
#                 log_probs = net(images)
#                 loss = self.loss_func(log_probs,labels)
#                 loss.backward() #this computes the gradient
#                 optimizer.step()
#                 batch_loss.append(loss.item())
#             epoch_loss.append(sum(batch_loss)/len(batch_loss))
#         return net,net.state_dict(),(sum(batch_loss)/len(batch_loss))


# def test_img2(net_g, datatest,bs,indexes,device=torch.device('cpu')):
#     net_g.eval()
#     # testing
#     test_loss = 0
#     correct = 0
#     data_loader = DataLoader(segmentdataset(datatest,indexes),batch_size=bs,shuffle=True)
    
#     for idx, (data, target) in enumerate(data_loader):
#         data = data.to(device)
#         target = target.to(device)
        
#         ## added this for radioML dataset
#         target = target.type(torch.long)
        
#         log_probs = net_g(data)
#         # sum up batch loss
#         test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
#         # get the index of the max log-probability
#         y_pred = log_probs.data.max(1, keepdim=True)[1]
#         correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

#     test_loss /= len(data_loader.dataset)
#     accuracy = 100*correct.item() / len(data_loader.dataset)
#     return accuracy, test_loss

if __name__ == '__main__':
    pwd = os.pardir
    cwd = os.getcwd()
    dtrain = RML(ldir=pwd+'/data/radio_ml/',train=True)
    dtest = RML(ldir=pwd+'/data/radio_ml/',train=False)
    
    # torch.manual_seed(1)
    # np.random.seed(1)
    # # # determine the number of datapoints per label
    # # train = {i: [] for i in range(10)}
    # # for index, label in enumerate(dtrain.y_data):
    # #     train[label].append(index)
        
    # # test = {i: [0] for i in range(10)} 
    # # for index, label in enumerate(dtest.y_data):
    # #     test[label].append(index)
    
    # # import matplotlib.pyplot as plt
    # # for i in range(10):
    # #     plt.figure()
    # #     plt.title('plot label '+str(i))
    # #     for j in range(3):
    # #         temp = dtrain[train[i][j]][0] # for concat of input data
    # #         temp = np.reshape(temp,(temp.shape[1]*temp.shape[2],temp.shape[0]))
    # #         plt.plot(temp)
    # #         # plt.plot(dtrain[train[i][j]][0])

    # device = torch.device('cuda')
    # d_in = 2*128
    # # d_in = 128
    # d_h = 64
    # d_out = 10
    # # global_net = MLP2(d_in,d_h,d_out).to(device)
    # global_net = CNNR(1,10).to(device)
    # # lr = 1e-1 #works better for MLP

    # for i in range(100):
    #     if i < 10:
    #         lr = 5e-2
    #     elif i < 20:
    #         lr = 1e-2
    #     elif i < 30:
    #         lr = 5e-3
    #     elif i < 40:
    #         1e-3
    #     elif i < 50:
    #         5e-4
    #     else:
    #         1e-4
        
    #     # if i < 10:
    #     #     lr = 1e-2
    #     # elif i < 20:
    #     #     lr = 5e-3
    #     # elif i < 30:
    #     #     lr = 1e-3
    #     # elif i < 40:
    #     #     5e-4
    #     # elif i < 50:
    #     #     1e-4
    #     # else:
    #     #     5e-5
        
    #     t_obj = LocalUpdate(device,bs=12,lr=lr,epochs=1,\
    #                 dataset=dtrain,indexes=range(dtrain.y_data.shape[0]))
    #     _,w,loss = t_obj.train(net=global_net)
    #     # print(w['layer_hidden.bias'])
    #     global_net.load_state_dict(w)
        
    #     c_acc, c_loss = \
    #         test_img2(global_net,dtest,\
    #         bs=12,indexes=range(dtest.y_data.shape[0]),device=device)
    
    #     print(c_acc,c_loss)




