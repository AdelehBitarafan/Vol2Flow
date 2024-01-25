"""
Vol2Flow
"""
# __________________________________________________________________________________ IMPORTS
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset, DataLoader
import copy
import skimage
from torch.utils import data

from Network import UnetReg
from Dataset import *
from Utils import *
from Losses import *
from Eval import *

# __________________________________________________________________________ ARGS and CONFIG
class Args:
    def __init__(self):
        
        self.model_description = 'Vol2Flow'
        self.image_volume_depth = 400
        self.image_shape = (128,128)
        self.size = 128
        self.lr = 1e-4
        self.epochs = 100
        self.batch_size = 1
        self.load_model = None
        self.M = 5
        self.loss ='mse' # mse, mae, ncc, ssim
        self.model_id = f'Vol2Flow_depth:{self.image_volume_depth}_M:{self.M}_{self.loss}'
        self.device = 'cuda'
        

        self.saving_base = 'models/'
        self.data_base = 'DATA/Training/'
        self.validation_data = 'DATA/Validation/SYNAPS/labeled_images.npy'
        
        self.model_dir = f'{self.saving_base}{self.model_id}/'
        

    def save_info(self):
        model_info = f'\n___ Model Info: ID {self.model_id} ___\n'
        model_info += f'___ model:        \n{self.model_description}\n'
        model_info += f'___ model_dir:  {self.model_dir} ___\n'
        model_info += f'lr:             {self.lr}\n'
        model_info += f'epochs:         {self.epochs}\n'
        model_info += f'batch_size:     {self.batch_size}\n'
        model_info += f'volume_depth:   {self.image_volume_depth}\n'
        model_info += f'image_shape:    {self.image_shape}\n'
        model_info += f'load_model:     {self.load_model}\n'
        model_info += f'M:              {self.M}\n'
  
        print(model_info)

        for obj in os.listdir(self.saving_base):
            if obj == str(self.model_id):
                print(f'ID Error: model with ID {self.model_id} already exists in {self.model_dir}')
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)

        with open(os.path.join(self.model_dir, f'model_{self.model_id}_info.txt'), 'w+') as info:
            info.write(model_info)
            info.close()

torch.manual_seed(10)
torch.cuda.manual_seed(10)
np.random.seed(10)
random.seed(10)
    
args = Args()
args.save_info()

# __________________________________________________________________________ CREATE NETWORK
class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size] 
        grids = torch.meshgrid(vectors) 
        grid  = torch.stack(grids) # y, x, z
        grid  = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow 

        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1) 
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1) 
            new_locs = new_locs[..., [2,1,0]]

        return F.grid_sample(src, new_locs, mode=self.mode)

torch.backends.cudnn.deterministic = True

enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16, 4]  # 4 added at the end to create registration field

inshape = (args.image_volume_depth, args.size, args.size)

model = UnetReg(
    inshape = inshape,
    infeats = 1,  # Number of input features (channels)
    nb_features=[enc_nf, dec_nf])

model.to(args.device).float()

model.to(args.device)
model.train()

single_slice_shape = torch.Size([args.size, args.size])
transformer = SpatialTransformer(single_slice_shape)
transformer = transformer.to(args.device)

# __________________________________________________________________________ LOSS FUNCTIONS
if args.loss=='mse':
    loss_func =  MSE().loss
elif args.loss=='ssim':
    loss_func = SSIM().loss
elif args.loss=='ncc':
    loss_func = NCC().loss
elif args.loss=='mae':
    loss_func = MAE().loss
    
# __________________________________________________________________________ OPTIMIZER
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# __________________________________________________________ LOAD SAVED MODEL (if provided)
if args.load_model is not None:
    checkpoint = torch.load(args.load_model)
    print(f'\n____ Loading model from {args.load_model} ____')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('\n_________ Model Loaded Successfully____\n')

# __________________________________________________________ TRAIN THE MODEL
losses_array, Dice_seg_array = [], []
the_best_dice = 0 
the_best_epoch = 0 
the_best_loss = 10000

import glob
train_files = glob.glob(args.data_base+'*.npy')


for epoch in range(1, args.epochs+1):
    print('Training the model ..... ')
    
    running_loss = []
    random.shuffle(train_files)
    for data_path in train_files:
        print(data_path)
        
        dataset = MyDataset(args.image_volume_depth, data_path, args.size)
        dataloader = torch.utils.data.DataLoader(dataset, args.batch_size)


        for _, data in enumerate(dataloader):
        
            volume = data
            inputs = volume.unsqueeze(1).to(args.device).float()
            
            _, e = valid_range(volume[0])
            idx_arr = np.arange(1, e-1)
            random.shuffle(idx_arr)
            
            for idx in idx_arr:
            
                flow = model(inputs).permute(2, 0, 1, 3, 4)
                
                start = max(idx - args.M, 0)
                end = min(idx+ args.M, args.image_volume_depth-1)

                volume_pred = torch.zeros(volume.shape)
                volume_pred[:, idx] = volume[:,idx].clone()
        
                max_ = min(args.image_volume_depth-1,idx+args.M)
                min_ = max(0, idx-args.M)

        
                '''
                forward
                '''
                for i in range(idx, max_, 1):
                    if volume[:,i+1].sum() <= 0:
                        break
                    volume_pred[:,i+1] = transformer(volume_pred[:,i:i+1].to(args.device).float().clone(), flow[i, :, :2])        
 
                '''
                backward
                '''
                for i in range(idx, min_, -1):
                    if volume[:,i-1].sum() <= 0:
                        break
                    volume_pred[:,i-1] = transformer(volume_pred[:,i:i+1].to(args.device).float().clone(), flow[i, :, 2:]) 


                loss = loss_func(volume[:,min_:max_+1].to(args.device).float(), volume_pred[:,min_:max_+1].to(args.device).float()) 
             
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                running_loss.append(loss.item())
    
    
    losses_array.append(np.mean(running_loss))
        
    # Testing ##############################
    
    print('Testing the model ..... ')
    model.eval()
    
    dice3D = evaluate(model, transformer, args.validation_data, args.device, args.size, args.image_volume_depth, ORGAN_IDX=6)
    
    Dice_seg_array.append(dice3D)
    
    print_training_log(epoch, args.epochs, np.mean(running_loss), dice3D, args.model_dir)
        
    if the_best_dice < dice3D:
        the_best_epoch = epoch
        the_best_dice = dice3D
        
        torch.save(model.state_dict(),  args.model_dir + 'BestModel_{}_{}'.format(epoch+1, dice3D))
        print('The model with the best dice is saved.')
        
        
    np.save(args.model_dir + '/epoch_Losses_training.npy', losses_array)
    np.save(args.model_dir + '/epoch_Dice_testing.npy', Dice_seg_array)


print('The best model is in epoch {} with {} dice score.'.format(the_best_epoch, the_best_dice))




