import os
import time
import numpy as np
import pickle
import json
import tqdm
import torch
import torch.nn as nn
import cv2
from torch.utils.data import Dataset, DataLoader
import copy
import skimage
from losses import adj_flow_loss
from torch.utils import data
from scipy.spatial.distance import dice, jaccard

from utils import *
import torch.nn.functional as F

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


def evaluate(model, data_path, device, organs):
    
    result_overall={}
    model = model.eval()
    
    with torch.set_grad_enabled(False):  
        
        dice_organs = []
        dice_organs_3d = []
        for orgn in organs:
        
            result_model=[]
            result_repeat=[]
            result_group={}
            decay = []
              
            validation_set = MyTestDataset(args.data_base, orgn)
            validation_generator = data.DataLoader(validation_set, batch_size=1, shuffle=False,
                                                num_workers=0, drop_last=False, collate_fn=my_collate)
            dice_images = []
            dice_images_3d = []
            for img_vol_norm, mask_truth, img_vol, pos_max, pos_start, pos_end, file_name in validation_generator:
                
                flow_all = model(img_vol.unsqueeze(0).float()).permute(2, 0, 1, 3, 4)
        
                file_name = file_name[0]
                pos_max = pos_max.squeeze().detach().cpu().numpy().item()
                pos_start = pos_start.squeeze().detach().cpu().numpy().item()
                pos_end = pos_end.squeeze().detach().cpu().numpy().item()
                mask_truth = mask_truth.squeeze().detach().cpu().numpy()
                
                # print(pos_max, pos_start, pos_end)
                
                dice_slices = []
                for pos_num, pos_current in enumerate([pos_max]):
                    
                    mask_pred = mask_truth.copy()
                    mask_pred *= 0
                    mask_pred[pos_current] = mask_truth[pos_current].copy()
                    time_count = []
                    
                    '''
                    backward
                    '''
                    vol_pred = []
                    vol_gt = []
                    
                    template = []
                    img_collection = (img_vol.squeeze().detach().cpu().numpy())[pos_current:pos_current+1][np.newaxis]
                    mask_collection = mask_truth[pos_current:pos_current+1].copy()
                    feat_collection = np.ones([])
                    temp=[]
                    for i in range(pos_current, 0, -1):                    
                        if mask_truth[i-1].sum() <= 0:
                            break


                        start_time = time.time()
                        flow = flow_all[pos_current, :, 2:]
                        _output = transformer(torch.from_numpy(mask_pred[pos_current:pos_current+1]).to(device).float().unsqueeze(0), flow)
                
                
                        output = (_output > 0.5) * 1.0
                        output = output.squeeze().detach().cpu().numpy()
                        
                        template = None
                        
                        output, template = verification_module(output, mask_collection, (img_vol.squeeze().detach().cpu().numpy())[i-1:i], img_collection, template)
                        
                        time_count.append(time.time() - start_time)

                        mask_pred[i-1] = output
                        vol_pred.append(mask_pred[i-1])
                        vol_gt.append(mask_truth[i-1])

                        dice_slices.append(dice_score(mask_truth[i-1], mask_pred[i-1]))
                        # print(f'{i-1}: {dice_score(mask_truth[i-1], mask_pred[i-1])}')
                        
                        mask_collection = np.concatenate((mask_collection, output[np.newaxis]), axis=0)
                        img_collection = np.concatenate((img_collection, (img_vol.squeeze().detach().cpu().numpy())[np.newaxis,i-1:i]), axis=0)
                        temp.append(1-dice(mask_pred[i-1].flatten(), mask_truth[i-1].flatten()))
                        
                    decay.append(temp) 
                    
                    vol_pred = vol_pred[::-1]
                    vol_gt = vol_gt[::-1]
                        
                    '''
                    forward
                    '''
                    template = []
                    img_collection = (img_vol.squeeze().detach().cpu().numpy())[pos_current:pos_current+1][np.newaxis]
                    mask_collection = mask_truth[pos_current:pos_current+1].copy()
                    feat_collection = np.ones([])
                    temp=[]
                    
                    vol_pred.append(mask_truth[pos_current])
                    vol_gt.append(mask_truth[pos_current])
                    
                    for i in range(pos_current, mask_pred.shape[0]-1, 1):
                        if mask_truth[i+1].sum() <= 0:
                            break
                            
                        start_time = time.time()
                        flow = flow_all[pos_current, :, :2]
                        _output = transformer(torch.from_numpy(mask_pred[pos_current:pos_current+1]).to(device).float().unsqueeze(0), flow)
  
                        output = (_output > 0.5) * 1.0
                        output = output.squeeze().detach().cpu().numpy()

                        template = None
                        
                        output, template = verification_module(output, mask_collection, (img_vol.squeeze().detach().cpu().numpy())[i+1:i+2], img_collection, template)
                        
                        time_count.append(time.time() - start_time)
                        
                        mask_pred[i+1] = output
                        vol_pred.append(mask_pred[i+1])
                        vol_gt.append(mask_truth[i+1])

                        dice_slices.append(dice_score(mask_truth[i+1], mask_pred[i+1]))
                        # print(f'{i+1}: {dice_score(mask_truth[i+1], mask_pred[i+1])}')
                        
                        mask_collection = np.concatenate((mask_collection, output[np.newaxis]), axis=0)
                        img_collection = np.concatenate((img_collection, (img_vol.squeeze().detach().cpu().numpy())[np.newaxis,i+1:i+2]), axis=0)
                        
                        temp.append(1-dice(mask_pred[i+1].flatten(), mask_truth[i+1].flatten()))
                        
                    decay.append(temp)
                    img_vol = img_vol.squeeze().detach().cpu().numpy()
                    
                    vol_pred = np.stack(vol_pred, 0)
                    vol_gt = np.stack(vol_gt, 0)
                    dice_3d = dice_score(vol_pred, vol_gt)
                    
                    print(f'image: {file_name} -- organ: {orgn} -- dice: {sum(dice_slices)/len(dice_slices)} -- dice3d: {dice_3d}')
                    dice_images.append(sum(dice_slices)/len(dice_slices))
                    dice_images_3d.append(dice_3d)
                    
                                
            dice_organs.append(sum(dice_images)/len(dice_images))
            dice_organs_3d.append(sum(dice_images_3d)/len(dice_images_3d))    
            #print(f'organ: {orgn} -- mean dice: {sum(dice_images)/len(dice_images)} -- mean dice 3d: {sum(dice_images_3d)/len(dice_images_3d)}')
            
        print(f'mean dice: {sum(dice_organs)/len(dice_organs)} -- mean dice 3d: {sum(dice_organs_3d)/len(dice_organs_3d)}')

        return dice_organs_3d


