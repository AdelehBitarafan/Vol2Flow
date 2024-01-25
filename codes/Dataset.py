import glob
import torch
import numpy as np
import random
from skimage.transform import radon, resize
from Utils import * 


    
class MyTestDataset(torch.utils.data.Dataset):
    
    def __init__(self, depth, path, size, organ):
        
        self.depth = depth
        self.size = size
        
        images = []
        labels = []
        
        images_loaded = np.load(path, allow_pickle=True)
        
        
        for i in range(len(images_loaded)):
            
            img = images_loaded[i].get('image')
            img = ((img - img.min()) / (img.max() - img.min()))
            
            lbl = images_loaded[i].get('label')
            lbl[np.where(lbl != organ)] = 0
            lbl[np.where(lbl == organ)] = 1
            lbl = lbl.astype('int8')
            
            
            images.append(img)  
            labels.append(lbl)
            
            
        print(len(images))
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        
        img = self.images[index]
        lbl = self.labels[index]
        
        
        if img.shape[0]< self.depth:
            img = np.concatenate([img, np.zeros((self.depth - len(img), *img.shape[1:])).astype('int16')], axis=0)
            lbl = np.concatenate([lbl, np.zeros((self.depth - len(lbl), *lbl.shape[1:])).astype('int8')], axis=0)
                
        else:
            img = img[img.shape[0] - self.depth:]
            lbl = lbl[lbl.shape[0] - self.depth:]

        x = []
        y = []
        for i in range(len(img)):
            x.append(resize(img[i], (self.size,self.size), anti_aliasing=True, preserve_range=True))
            y.append(resize(lbl[i], (self.size,self.size), anti_aliasing=True, preserve_range=True))
        
        img = np.asarray(x)
        lbl = np.asarray(y)

        pos_max = np.argmax(lbl.sum((1, 2)))
        pos = np.nonzero(np.sum(lbl, (1,2)))[0]
        
        return img, lbl, pos_max, pos[0], pos[-1]
    
    
    
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, depth, path, size):
        
        self.depth = depth
        self.size = size
        
        images = []
        
        images_loaded = np.load(path, allow_pickle=True)

        for i in range(len(images_loaded)):
            img = images_loaded[i].get('image')
            
            img = ((img - img.min()) / (img.max() - img.min()))
            images.append(img)    

        
        print(len(images))
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_3d_sample = self.images[index]
        
        resize_size = random.randint(self.size+1, self.size+50)
        x = np.random.randint(0, resize_size - self.size)
        y = np.random.randint(0, resize_size - self.size)
        
        img = []
        for z1 in range(len(image_3d_sample)):
            frame1 = image_3d_sample[z1, :, :]
            frame1 = resize(frame1, (resize_size, resize_size), anti_aliasing=True, preserve_range=True)
            [frame1,frame1] = get_random_crop([frame1,frame1], self.size, self.size, x, y)  
            
            img.append(frame1)
         
        
        image_3d_sample = np.asarray(img)

        
        if len(image_3d_sample)<self.depth:
            
            image_3d_sample = np.concatenate([image_3d_sample, np.zeros((self.depth - len(image_3d_sample), *image_3d_sample.shape[1:])).astype('int16')], axis=0)
            
            return image_3d_sample
         
        else:
            start = np.random.randint(0, len(image_3d_sample)-self.depth+1)
            image_3d_sample = image_3d_sample[start:start+self.depth]
            
            return image_3d_sample


