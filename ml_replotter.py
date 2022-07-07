# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 10:01:17 2021

@author: henry
"""
import os 
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import string

# proof of concept replotter

mpl.style.use('default')
mpl.rc('font',family='Times New Roman')
csfont = {'fontname':'Times New Roman'}
markersize = dict(markersize=2)

cwd = os.getcwd()

save_loc = cwd+'/ml_plots/'
data_loc = cwd+'/data/'

data_source = 'mnist'
data_source = 'fmnist'
data_source = 'cifar10'
data_source = 'mlradio'
lwd = 2.5

# %% plot 2 ratio plots with taus1 = 1
# static ratio - 1,1 at taus1=taus2=2

total_fl_ratios = []
total_pfl_ratios = []

full_fl_ratios = []
full_pfl_ratios = []

full_fl_loss_ratios = []
full_pfl_loss_ratios = []

## initial values
init_acc = 10.5
init_loss = 2.302232319

nn_style = 'CNN'
# nn_style = 'CNN2'
# nn_style = 'MLP'
ratio_vec = [1,2,4,8]

# tseed = None
tseed = 1
# tseed = 2
for ratio in [1,2,4,8]:
    ## reload data
    total_fl_accs, total_pfl_accs = [], []
    full_fl_accs, full_pfl_accs = [], []
    
    fl_loss, pfl_loss = [], []
    full_fl_loss, full_pfl_loss = [], []
    
    swarm_period = 1
    global_period = ratio*swarm_period
    for iid_style in ['mild']:#['mild']:  'iid'
        # ## personalized accuracies
        # with open(data_loc+'3fl_acc_'+iid_style+'_'+str(ratio)+'_'+data_source \
        #           +'_'+str(swarm_period)+'_'+str(global_period)+'_'+nn_style+\
        #         '_tseed'+str(tseed)+'_debug','rb') as f:
        #     fl_acc = pk.load(f)
        
        # with open(data_loc+'3hn_pfl_acc_'+iid_style+'_'+str(ratio)+'_'+data_source \
        #             +'_'+str(swarm_period)+'_'+str(global_period)+'_'+nn_style+\
        #         '_tseed'+str(tseed)+'_debug','rb') as f:
        #     pfl_acc = pk.load(f)
        
        ## full (i.e. global) accuracies
        with open(data_loc+'3full_fl_acc_'+iid_style+'_'+str(ratio)+'_'+data_source \
                  +'_'+str(swarm_period)+'_'+str(global_period)+'_'+nn_style+\
                '_tseed'+str(tseed)+'_debug','rb') as f:
            f_fl_acc = pk.load(f)        

        
        with open(data_loc+'3full_hn_pfl_acc_'+iid_style+'_'+str(ratio)+'_'+data_source \
                    +'_'+str(swarm_period)+'_'+str(global_period)+'_'+nn_style+\
                '_tseed'+str(tseed)+'_debug','rb') as f:
            f_pfl_acc = pk.load(f)
            
        full_fl_accs.append(f_fl_acc)
        full_pfl_accs.append(f_pfl_acc)
        
        ## full/global losses
        with open(data_loc+'3full_fl_loss_'+iid_style+'_'+str(ratio)+'_'+data_source \
                  +'_'+str(swarm_period)+'_'+str(global_period)+'_'+nn_style+\
                '_tseed'+str(tseed)+'_debug','rb') as f:
            f_fl_loss = pk.load(f)        
        
        with open(data_loc+'3full_hn_pfl_loss_'+iid_style+'_'+str(ratio)+'_'+data_source \
                  +'_'+str(swarm_period)+'_'+str(global_period)+'_'+nn_style+\
                '_tseed'+str(tseed)+'_debug','rb') as f:
            f_pfl_loss = pk.load(f)
        
        full_fl_loss.append(f_fl_loss)
        full_pfl_loss.append(f_pfl_loss)
        
        # total_fl_accs.append(fl_acc)
        # total_pfl_accs.append(pfl_acc)
    
    full_fl_loss_ratios.append(full_fl_loss)
    full_pfl_loss_ratios.append(full_pfl_loss)
    
    # total_fl_ratios.append(total_fl_accs)
    # total_pfl_ratios.append(total_pfl_accs)
    
    full_fl_ratios.append(full_fl_accs)
    full_pfl_ratios.append(full_pfl_accs)
    
plt.figure(2)
f2,ax2 = plt.subplots(1,2,figsize=(10,3.3),dpi=100)#,sharey=True)


## temporary loss plotter
ind = 0
for i in range(len(ratio_vec)): #3
    if i == 0:
        temp_indexes = np.arange(0,40,step=1) #total_fl_ratios[i][0]        
    elif i == 1:
        temp_indexes = np.arange(0,40,step=2)
    else:
        temp_indexes = np.arange(0,40,step=4)
        
    temp_indexes = np.arange(0,40,step=ratio_vec[i])
    temp_indexes += ratio_vec[i]
    temp_indexes = list(temp_indexes)
    temp_indexes.append(0)
    temp_indexes = sorted(temp_indexes)    

    full_fl_loss_ratios[i][0].insert(0,init_loss)
    full_pfl_loss_ratios[i][0].insert(0,init_loss)
    
    if i == 0:        #+r'$\tau^{\mathsf{L}} = 1$ '+
        ax2[ind].plot(temp_indexes,full_fl_loss_ratios[i][0],\
            label='H-FL '+ r'$\tau^{\mathsf{G}}_s = 1$',\
            color='darkgoldenrod',linestyle='solid',linewidth=lwd)
        ax2[ind].plot(temp_indexes,full_pfl_loss_ratios[i][0],\
            label='HN-PFL ' + r'$\tau^{\mathsf{G}}_s = 1$',\
            color='black',linestyle='solid',linewidth=lwd)
    elif i == 1:
        ax2[ind].plot(temp_indexes,full_fl_loss_ratios[i][0],\
            label='H-FL ' + r'$\tau^{\mathsf{G}}_s = 2$',\
            color='darkgoldenrod',linestyle='dashed',linewidth=lwd)
        ax2[ind].plot(temp_indexes,full_pfl_loss_ratios[i][0],\
            label='HN-PFL ' + r'$\tau^{\mathsf{G}}_s = 2$',\
            color='black',linestyle='dashed',linewidth=lwd)
    elif i ==2:
        ax2[ind].plot(temp_indexes,full_fl_loss_ratios[i][0],\
            label='H-FL ' + r'$\tau^{\mathsf{G}}_s = 4$',\
            color='darkgoldenrod',linestyle='dotted',linewidth=lwd)
        ax2[ind].plot(temp_indexes,full_pfl_loss_ratios[i][0],\
            label='HN-PFL '+ r'$\tau^{\mathsf{G}}_s = 4$',\
            color='black',linestyle='dotted',linewidth=lwd)
    else: 
        ax2[ind].plot(temp_indexes,full_fl_loss_ratios[i][0],\
            label='H-FL '+ r'$\tau^{\mathsf{G}}_s = 8$',\
            color='darkgoldenrod',linestyle='dashdot',linewidth=lwd)
        ax2[ind].plot(temp_indexes,full_pfl_loss_ratios[i][0],\
            label='HN-PFL '+ r'$\tau^{\mathsf{G}}_s = 8$',\
            color='black',linestyle='dashdot',linewidth=lwd)

ax2[ind].set_title('a) '+data_source.upper()+' Classification Loss',y=-0.3,fontsize=14)
ax2[ind].set_ylabel('Log Loss',fontsize=13)
ax2[ind].set_xlabel('Local Iteration',fontsize=13)

# txlab = ax2[ind].get_xticklabels()
# txlab = [i.get_text() for i in txlab]
# tylab = ax2[ind].get_yticklabels()
if data_source == 'mnist':
    tylab = ['1.5', '1.6', '1.7', '1.8', '1.9', '2.0', '2.1', '2.2', '2.3', '2.4']
elif data_source == 'fmnist':
    tylab = ['1.8', '1.9', '2.0', '2.1', '2.2', '2.3', '2.4']
elif data_source == 'cifar10':
    tylab = ['1.2','1.4','1.6','1.8','2.0','2.2']
elif data_source == 'mlradio':
    tylab = ['0.5','0.75','1.00','1.25','1.50','1.75','2.00','2.25']

txlab = ['−10', '0', '10', '20', '30', '40', '50']
ax2[ind].set_xticklabels(txlab,fontsize=12)
ax2[ind].set_yticklabels(tylab,fontsize=12)

ax2[ind].grid(True)
ax2[ind].legend()


ind = 1
for i in range(len(ratio_vec)): 
    if i == 0:
        temp_indexes = np.arange(0,40,step=1)#)4) #total_fl_ratios[i][0]
    elif i == 1:
        temp_indexes = np.arange(0,40,step=2)#40/6)
    else:
        temp_indexes = np.arange(0,40,step=4)#8)
    # temp_indexes = np.arange(0,10,step=1)
    
    temp_indexes = np.arange(0,40,step=ratio_vec[i])
    temp_indexes += ratio_vec[i]
    temp_indexes = list(temp_indexes)
    temp_indexes.append(0)
    temp_indexes = sorted(temp_indexes)        
    
    full_fl_ratios[i][0].insert(0,init_acc)
    full_pfl_ratios[i][0].insert(0,init_acc)

    if i == 0:
        ax2[ind].plot(temp_indexes,full_fl_ratios[i][0],\
            label='H-FL '+r'$\tau^{\mathsf{L}} = 1$ '+ r'$\tau^{\mathsf{G}} = 1$',\
            color='darkgoldenrod',linestyle='solid',linewidth=lwd)#,marker='x')
        ax2[ind].plot(temp_indexes,full_pfl_ratios[i][0],\
            label='HN-PFL '+r'$\tau^{\mathsf{L}} = 1$ '+ r'$\tau^{\mathsf{G}} = 1$',\
            color='black',linestyle='solid',linewidth=lwd)#,marker='x')
    elif i == 1:
        ax2[ind].plot(temp_indexes,full_fl_ratios[i][0],\
            label='H-FL '+r'$\tau_{s}^{\mathsf{L}} = 1$ '+ r'$\tau^{\mathsf{G}} = 2$',\
            color='darkgoldenrod',linestyle='dashed',linewidth=lwd)#,marker='x')
        ax2[ind].plot(temp_indexes,full_pfl_ratios[i][0],\
            label='HN-PFL '+r'$\tau_{s}^{\mathsf{L}} = 1$ '+ r'$\tau^{\mathsf{G}} = 2$',\
            color='black',linestyle='dashed',linewidth=lwd)#,marker='x')
    elif i == 2:
        ax2[ind].plot(temp_indexes,full_fl_ratios[i][0],\
            label='H-FL '+r'$\tau^{\mathsf{L}} = 1$ '+ r'$\tau^{\mathsf{G}} = 4$',\
            color='darkgoldenrod',linestyle='dotted',linewidth=lwd)#,marker='x')
        ax2[ind].plot(temp_indexes,full_pfl_ratios[i][0],\
            label='HN-PFL '+r'$\tau^{\mathsf{L}} = 1$ '+ r'$\tau^{\mathsf{G}} = 4$',\
            color='black',linestyle='dotted',linewidth=lwd)#,marker='x')
    else:
        ax2[ind].plot(temp_indexes,full_fl_ratios[i][0],\
            label='H-FL '+r'$\tau^{\mathsf{L}} = 1$ '+ r'$\tau^{\mathsf{G}} = 8$',\
            color='darkgoldenrod',linestyle='dashdot',linewidth=lwd)#,marker='x')
        ax2[ind].plot(temp_indexes,full_pfl_ratios[i][0],\
            label='HN-PFL '+r'$\tau^{\mathsf{L}} = 1$ '+ r'$\tau^{\mathsf{G}} = 8$',\
            color='black',linestyle='dashdot',linewidth=lwd)#,marker='x')


ax2[ind].set_title('b) '+data_source.upper()+' Classification Accuracy',y=-0.3,fontsize=14)

if data_source == 'mnist':
    tylab = ['0', '20', '40', '60', '80', '100']
elif data_source =='fmnist':
    tylab = ['0', '10', '20', '30', '40', '50', '60', '70']
elif data_source == 'cifar10':
    tylab = ['0','10','20','30','40','50']
elif data_source == 'mlradio':
    tylab = ['0','20','40','60','80'] 
    
ax2[ind].set_xticklabels(txlab,fontsize=12)
ax2[ind].set_yticklabels(tylab,fontsize=12)

ax2[ind].set_ylabel('Classification Accuracy (%)',fontsize=13)
ax2[ind].set_xlabel('Local Iteration',fontsize=13)
ax2[ind].grid(True)



h,l = ax2[0].get_legend_handles_labels()
kw = dict(ncol=4,loc = 'lower center',frameon=False)
kw2 = dict(ncol=4,loc = 'lower center',frameon=False)
#(x, y, width, height)
leg1 = ax2[0].legend(h[1::2],l[1::2],bbox_to_anchor=(-0.1,1.1,2.2,0.2),\
                       mode='expand',fontsize='large',**kw2)
leg2 = ax2[0].legend(h[0::2],l[0::2],bbox_to_anchor=(-0.1,1.02,2.2,0.2),\
                       mode='expand',fontsize='large',**kw)
ax2[0].add_artist(leg1)
plt.subplots_adjust(top=0.8,wspace=0.15,hspace=0.15)

# plt.savefig(cwd+'/ml_plots/mild_'+data_source+'_ovr_swarms.pdf',bbox_inches='tight')
# plt.savefig(cwd+'/ml_plots/mild_'+data_source+'_ovr_swarms_temp.pdf',bbox_inches='tight')


# %% plot 3 ratio plots with taus2 = 1


total_fl_ratios = []
total_pfl_ratios = []

full_fl_ratios = []
full_pfl_ratios = []

full_fl_loss_ratios = []
full_pfl_loss_ratios = []

## initial values
init_acc = 10.5
init_loss = 2.302232319

# data_source = 'mnist'
# data_source = 'fmnist'

# nn_style = 'CNN2'
# nn_style = 'MLP'
nn_style = 'CNN'
ratio_vec = [1,2,4,8]
for ratio in [1,2,4,8]:
    ## reload data
    total_fl_accs, total_pfl_accs = [], []
    full_fl_accs, full_pfl_accs = [], []
    
    fl_loss, pfl_loss = [], []
    full_fl_loss, full_pfl_loss = [], []
    
    global_period = 1
    swarm_period = ratio*global_period
    for iid_style in ['mild']:#['mild']: 'iid'
        
        # ## personalized accuracies
        # with open(data_loc+'3fl_acc_'+iid_style+'_'+str(ratio)+'_'+data_source \
        #           +'_'+str(swarm_period)+'_'+str(global_period)+'_'+nn_style+\
        #         '_tseed'+str(tseed)+'_debug','rb') as f:
        #     fl_acc = pk.load(f)
        
        # with open(data_loc+'3hn_pfl_acc_'+iid_style+'_'+str(ratio)+'_'+data_source \
        #             +'_'+str(swarm_period)+'_'+str(global_period)+'_'+nn_style+\
        #         '_tseed'+str(tseed)+'_debug','rb') as f:
        #     pfl_acc = pk.load(f)
        
        ## full (i.e. global) accuracies
        with open(data_loc+'3full_fl_acc_'+iid_style+'_'+str(ratio)+'_'+data_source \
                  +'_'+str(swarm_period)+'_'+str(global_period)+'_'+nn_style+\
                '_tseed'+str(tseed)+'_debug','rb') as f:
            f_fl_acc = pk.load(f)        
        
        with open(data_loc+'3full_hn_pfl_acc_'+iid_style+'_'+str(ratio)+'_'+data_source \
                    +'_'+str(swarm_period)+'_'+str(global_period)+'_'+nn_style+\
                '_tseed'+str(tseed)+'_debug','rb') as f:
            f_pfl_acc = pk.load(f)
            
        full_fl_accs.append(f_fl_acc)
        full_pfl_accs.append(f_pfl_acc)
        
        ## full/global losses
        with open(data_loc+'3full_fl_loss_'+iid_style+'_'+str(ratio)+'_'+data_source \
                  +'_'+str(swarm_period)+'_'+str(global_period)+'_'+nn_style+\
                '_tseed'+str(tseed)+'_debug','rb') as f:
            f_fl_loss = pk.load(f)        
        
        with open(data_loc+'3full_hn_pfl_loss_'+iid_style+'_'+str(ratio)+'_'+data_source \
                  +'_'+str(swarm_period)+'_'+str(global_period)+'_'+nn_style+\
                '_tseed'+str(tseed)+'_debug','rb') as f:
            f_pfl_loss = pk.load(f)
        
        full_fl_loss.append(f_fl_loss)
        full_pfl_loss.append(f_pfl_loss)
        
        # total_fl_accs.append(fl_acc)
        # total_pfl_accs.append(pfl_acc)
    
    full_fl_loss_ratios.append(full_fl_loss)
    full_pfl_loss_ratios.append(full_pfl_loss)
    
    # total_fl_ratios.append(total_fl_accs)
    # total_pfl_ratios.append(total_pfl_accs)
    
    full_fl_ratios.append(full_fl_accs)
    full_pfl_ratios.append(full_pfl_accs)
    
plt.figure(2)
f2,ax2 = plt.subplots(1,2,figsize=(10,3.3),dpi=100)#,sharey=True)

## temporary loss plotter
ind = 0
for i in range(len(ratio_vec)): #3
    if i == 0:
        temp_indexes = np.arange(0,40,step=1) #total_fl_ratios[i][0]        
    elif i == 1:
        temp_indexes = np.arange(0,40,step=2)
    else:
        temp_indexes = np.arange(0,40,step=4)
        
    temp_indexes = np.arange(0,40,step=ratio_vec[i])
    temp_indexes += ratio_vec[i]
    temp_indexes = list(temp_indexes)
    temp_indexes.append(0)
    temp_indexes = sorted(temp_indexes)    

    full_fl_loss_ratios[i][0].insert(0,init_loss)
    full_pfl_loss_ratios[i][0].insert(0,init_loss)

    if i == 0: #+ r'$\tau^{\mathsf{G}} = 1$'
        ax2[ind].plot(temp_indexes,full_fl_loss_ratios[i][0],\
            label='H-FL '+r'$\tau^{\mathsf{L}}_s = 1$ ',\
            color='darkgoldenrod',linestyle='solid',linewidth=lwd)
        ax2[ind].plot(temp_indexes,full_pfl_loss_ratios[i][0],\
            label='HN-PFL '+r'$\tau^{\mathsf{L}}_s = 1$ ',\
            color='black',linestyle='solid',linewidth=lwd)        
    elif i == 1:
        ax2[ind].plot(temp_indexes,full_fl_loss_ratios[i][0],\
            label='H-FL '+r'$\tau^{\mathsf{L}}_s = 2$ ',\
            color='darkgoldenrod',linestyle='dashed',linewidth=lwd)
        ax2[ind].plot(temp_indexes,full_pfl_loss_ratios[i][0],\
            label='HN-PFL '+r'$\tau^{\mathsf{L}}_s = 2$ ',\
            color='black',linestyle='dashed',linewidth=lwd)   
    elif i ==2:
        ax2[ind].plot(temp_indexes,full_fl_loss_ratios[i][0],\
            label='H-FL '+r'$\tau^{\mathsf{L}}_s = 4$ ',\
            color='darkgoldenrod',linestyle='dotted',linewidth=lwd)
        ax2[ind].plot(temp_indexes,full_pfl_loss_ratios[i][0],\
            label='HN-PFL '+r'$\tau^{\mathsf{L}}_s = 4$ ',\
            color='black',linestyle='dotted',linewidth=lwd)
    else: 
        ax2[ind].plot(temp_indexes,full_fl_loss_ratios[i][0],\
            label='H-FL '+r'$\tau^{\mathsf{L}}_s = 8$ ',\
            color='darkgoldenrod',linestyle='dashdot',linewidth=lwd)
        ax2[ind].plot(temp_indexes,full_pfl_loss_ratios[i][0],\
            label='HN-PFL '+r'$\tau^{\mathsf{L}}_s = 8$ ',\
            color='black',linestyle='dashdot',linewidth=lwd)

ax2[ind].set_title('a) '+data_source.upper()+' Classification Loss',y=-0.3,fontsize=14)
ax2[ind].set_ylabel('Log Loss',fontsize=13)
ax2[ind].set_xlabel('Local Iteration',fontsize=13)

if data_source == 'mnist':
    tylab = ['1.5', '1.6', '1.7', '1.8', '1.9', '2.0', '2.1', '2.2', '2.3', '2.4']
elif data_source =='fmnist':
    tylab = ['1.8', '1.9', '2.0', '2.1', '2.2', '2.3', '2.4']
elif data_source == 'cifar10':
    tylab = ['1.0','1.2','1.4','1.6','1.8','2.0','2.2'] 
elif data_source == 'mlradio':
    tylab = ['0.5','0.75','1.00','1.25','1.50','1.75','2.00','2.25']

ax2[ind].set_xticklabels(txlab,fontsize=12)
ax2[ind].set_yticklabels(tylab,fontsize=12)

ax2[ind].grid(True)
ax2[ind].legend()


ind = 1
for i in range(len(ratio_vec)): 
    if i == 0:
        temp_indexes = np.arange(0,40,step=1)#)4) #total_fl_ratios[i][0]
    elif i == 1:
        temp_indexes = np.arange(0,40,step=2)#40/6)
    else:
        temp_indexes = np.arange(0,40,step=4)#8)
    # temp_indexes = np.arange(0,10,step=1)
    
    temp_indexes = np.arange(0,40,step=ratio_vec[i])
    temp_indexes += ratio_vec[i]
    temp_indexes = list(temp_indexes)
    temp_indexes.append(0)
    temp_indexes = sorted(temp_indexes)        
    
    full_fl_ratios[i][0].insert(0,init_acc)
    full_pfl_ratios[i][0].insert(0,init_acc)

    if i == 0:
        ax2[ind].plot(temp_indexes,full_fl_ratios[i][0],\
            label='H-FL '+r'$\tau^{\mathsf{L}}_s = 1$ ',\
            color='darkgoldenrod',linestyle='solid',linewidth=lwd)#,marker='x')
        ax2[ind].plot(temp_indexes,full_pfl_ratios[i][0],\
            label='HN-PFL '+r'$\tau^{\mathsf{L}}_s = 1$ ',\
            color='black',linestyle='solid',linewidth=lwd)#,marker='x')
    elif i == 1:
        ax2[ind].plot(temp_indexes,full_fl_ratios[i][0],\
            label='H-FL '+r'$\tau^{\mathsf{L}}_s = 2$ ',\
            color='darkgoldenrod',linestyle='dashed',linewidth=lwd)#,marker='x')
        ax2[ind].plot(temp_indexes,full_pfl_ratios[i][0],\
            label='HN-PFL '+r'$\tau^{\mathsf{L}}_s = 2$ ',\
            color='black',linestyle='dashed',linewidth=lwd)#,marker='x')
    elif i == 2:
        ax2[ind].plot(temp_indexes,full_fl_ratios[i][0],\
            label='H-FL '+r'$\tau^{\mathsf{L}}_s = 4$ ',\
            color='darkgoldenrod',linestyle='dotted',linewidth=lwd)#,marker='x')
        ax2[ind].plot(temp_indexes,full_pfl_ratios[i][0],\
            label='HN-PFL '+r'$\tau^{\mathsf{L}}_s = 4$ ',\
            color='black',linestyle='dotted',linewidth=lwd)#,marker='x')
    else:
        ax2[ind].plot(temp_indexes,full_fl_ratios[i][0],\
            label='H-FL '+r'$\tau^{\mathsf{L}}_s = 8$ ',\
            color='darkgoldenrod',linestyle='dashdot',linewidth=lwd)#,marker='x')
        ax2[ind].plot(temp_indexes,full_pfl_ratios[i][0],\
            label='HN-PFL '+r'$\tau^{\mathsf{L}}_s = 8$ ',\
            color='black',linestyle='dashdot',linewidth=lwd)#,marker='x')

ax2[ind].set_title('b) '+data_source.upper()+' Classification Accuracy',y=-0.3,fontsize=14)
ax2[ind].set_ylabel('Classification Accuracy (%)',fontsize=13)
ax2[ind].set_xlabel('Local Iteration',fontsize=13)

if data_source == 'mnist':
    tylab = ['0', '20', '40', '60', '80', '100']
elif data_source == 'fmnist':
    tylab = ['0', '10', '20', '30', '40', '50', '60', '70']
elif data_source == 'cifar10':
    tylab = ['0','10','20','30','40','50']
elif data_source == 'mlradio':
    tylab = ['0','20','40','60','80'] 
    
ax2[ind].set_xticklabels(txlab,fontsize=12)
ax2[ind].set_yticklabels(tylab,fontsize=12)

ax2[ind].grid(True)


h,l = ax2[0].get_legend_handles_labels()
kw = dict(ncol=4,loc = 'lower center',frameon=False)
kw2 = dict(ncol=4,loc = 'lower center',frameon=False)
#(x, y, width, height)
leg1 = ax2[0].legend(h[1::2],l[1::2],bbox_to_anchor=(-0.1,1.1,2.2,0.2),\
                       mode='expand',fontsize='large',**kw2)
leg2 = ax2[0].legend(h[0::2],l[0::2],bbox_to_anchor=(-0.1,1.02,2.2,0.2),\
                       mode='expand',fontsize='large',**kw)
ax2[0].add_artist(leg1)
plt.subplots_adjust(top=0.8,wspace=0.15,hspace=0.15)

# plt.savefig(cwd+'/ml_plots/mild_'+data_source+'_ovr_global.pdf',bbox_inches='tight')
# plt.savefig(cwd+'/ml_plots/mild_'+data_source+'_ovr_global_temp.pdf',bbox_inches='tight')

