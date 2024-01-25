import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import skimage
from torch.utils import data
from scipy.spatial.distance import dice, jaccard
import torch.nn.functional as F
from skimage.transform import radon, resize
from Utils import *

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

def Mask_Propagation(model, transformer, images, labels, mid_lbl_idx, device = 'cuda' , threshold = 0.5):
    
    in_shape = (1, 1, images.shape[1], images.shape[1])
    zero = torch.zeros(in_shape).to(device)
    one = torch.ones(in_shape).to(device)
    
    depth = images.shape[0]
    
    predicted_lbl = np.zeros((images.shape))
    
    with torch.no_grad():
        inputs = torch.from_numpy(images[np.newaxis, np.newaxis]).to(device).float()
        flow_all = model(inputs).permute(2, 0, 1, 3, 4)
        
        range_obj_by_direction = {'forward': range(mid_lbl_idx+1, depth),'backward': range(mid_lbl_idx-1, -1, -1)} 
        
        for direction in ('forward', 'backward'):

            source_lbl = torch.from_numpy(labels[mid_lbl_idx]).to(device).float().unsqueeze(0).unsqueeze(0)
                      
            predicted_lbl[mid_lbl_idx] = labels[mid_lbl_idx]
            
            range_obj = range_obj_by_direction.get(direction)
               
            img_collection = images[mid_lbl_idx][np.newaxis][np.newaxis]
            mask_collection = labels[mid_lbl_idx][np.newaxis].copy()


            for target_idx in range_obj:

                target_lbl = torch.from_numpy(labels[target_idx]).to(device).float().unsqueeze(0).unsqueeze(0)
                          
                if target_lbl.sum()==0: 
                    break

                if direction=='forward':
                    flow = flow_all[target_idx-1, :, :2]
                
                elif direction=='backward':
                    flow = flow_all[target_idx+1, :, 2:]


                    
                moved_lbl = transformer(source_lbl, flow)  
                moved_lbl = torch.where(moved_lbl >= threshold, one, zero)   
                


                moved_lbl = moved_lbl.squeeze().detach().cpu().numpy()
                template = None
                if direction=='forward':
                    moved_lbl, template = verification_module(moved_lbl, mask_collection, images[target_idx-1][np.newaxis], img_collection, template)
                    img_collection = np.concatenate((img_collection, images[target_idx-1][np.newaxis][np.newaxis]), axis=0)

                if direction=='backward':
                    moved_lbl, template = verification_module(moved_lbl, mask_collection, images[target_idx+1][np.newaxis], img_collection, template)
                    img_collection = np.concatenate((img_collection, images[target_idx+1][np.newaxis][np.newaxis]), axis=0)


                mask_collection = np.concatenate((mask_collection, moved_lbl[np.newaxis]), axis=0)
                moved_lbl = torch.from_numpy(moved_lbl).to(device).float().unsqueeze(0).unsqueeze(0)
                    
                    
                source_lbl = moved_lbl
                predicted_lbl[target_idx] = moved_lbl.squeeze().detach().cpu().numpy()
  

    return predicted_lbl


def evaluate(model, transformer, path, device, size, depth, ORGAN_IDX):
    images_loaded = np.load(path , allow_pickle=True)

    val_images = []
    val_labels = []


    for j in range(len(images_loaded)):

        lbl = images_loaded[j].get('label')
        lbl[np.where(lbl != ORGAN_IDX)] = 0
        lbl[np.where(lbl == ORGAN_IDX)] = 1
        lbl = lbl.astype('int16')
    
    
        img = images_loaded[j].get('image')
        img = ((img - img.min()) / (img.max() - img.min()))

    
        x = []
        y = []
        for i in range(len(img)):
            x.append(resize(img[i], (size, size), anti_aliasing=True, preserve_range=True))
            y.append(resize(lbl[i], (size, size), anti_aliasing=True, preserve_range=True))
            
        val_images.append(np.asarray(x))
        val_labels.append(np.asarray(y))

    print('Validation Data:  ', len(val_images), val_images[0].shape, val_labels[0].shape)



    model.eval()
    dice_lbl = []

    for i in range(len(val_images)):

        Large_slice = get_LargeSlice(val_labels[i])
    
        start = max(0, Large_slice-int(depth/2))
        end = min(len(val_labels[i]), Large_slice+int(depth/2))
    
    
        val_labels_subvol = val_labels[i][start:end].copy()
        val_images_subvol = val_images[i][start:end].copy()
    
        if len(val_images_subvol)<depth and start==0:
        
            diff = depth - len(val_images_subvol)
            val_labels_subvol = val_labels[i][start:end+diff].copy()
            val_images_subvol = val_images[i][start:end+diff].copy()
        
        
        if len(val_images_subvol)<depth and end==len(val_labels[i]):
        
            diff = depth - len(val_images_subvol)
            val_labels_subvol = val_labels[i][start-diff:end].copy()
            val_images_subvol = val_images[i][start-diff:end].copy()
        
        Large_slice = get_LargeSlice(val_labels_subvol)
        start,end = valid_range(val_labels_subvol)
    
        predicted = Mask_Propagation(model, transformer, val_images_subvol, val_labels_subvol, Large_slice, device)

        dice3D_= dice_volume(predicted[start:end],val_labels_subvol[start:end])
  
        dice_lbl.append(dice3D_)
        print(i, dice3D_)
                    
    return np.mean(dice_lbl)



