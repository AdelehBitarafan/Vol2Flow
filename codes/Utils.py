import numpy as np
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes, binary_erosion
from scipy import ndimage
    

def get_random_crop(image_list, crop_height, crop_width, x, y):

    crop = [image[y: y + crop_height, x: x + crop_width] for image in image_list]
    return crop


def verification_module(mask_img, mask_collection, img, img_collection, template):
    
    def getLargestCC(segmentation):
        labels = label(segmentation)
        if labels.max() == 0:
            return segmentation
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        return (largestCC>0).astype(int)
    
    def refine(segmentation, original):
        chull = convex_hull_image(segmentation)
        return segmentation#chull#*original
    
    # print(mask_img.shape, mask_collection.shape, img.shape, img_collection.shape)
    
    '''
    mask_img - (h, w)
    mask_collection - (f, h, w)
    img - (c, h, w)
    img_collection - (f, c, h, w)
    '''
    
    num_class = int(np.max(mask_collection[-1]))
    mask_refine = np.zeros(mask_img.shape)
    
    
    for nc in range(1,num_class+1,1):
        
        mask_history = (mask_collection[-1]==nc).astype(float)
        mask_img_temp = (mask_img==nc).astype(float)

        mask_neg = binary_dilation(mask_history, iterations=5)-mask_history
        mask_neg = np.tile(np.expand_dims(mask_neg,0), (img.shape[0],1,1))
        mask_pos = np.tile(np.expand_dims(mask_history,0), (img.shape[0],1,1)) # c, h, w
        
        feature_positive=np.ones(img.shape)*np.tile(np.array([np.mean(img_collection[-1,i][mask_pos[i]==1]) for i in range(mask_pos.shape[0])])[:,np.newaxis, np.newaxis], (1,img.shape[-2],img.shape[-1]))
        feature_negative=np.ones(img.shape)*np.tile(np.array([np.mean(img_collection[-1,i][mask_neg[i]==1]) for i in range(mask_neg.shape[0])])[:,np.newaxis, np.newaxis], (1,img.shape[-2],img.shape[-1]))
        feature_positive = np.sum((img-feature_positive)**2,axis=0)
        feature_negative = np.sum((img-feature_negative)**2,axis=0)
        
        positive_likelihood = np.zeros(feature_positive.shape)
        positive_likelihood[feature_positive<feature_negative] = 1
        positive_likelihood = positive_likelihood*mask_img_temp # + (binary_erosion(mask_img_temp, iterations=5))
        positive_likelihood[positive_likelihood>0]=1
        
        positive_likelihood[~np.isfinite(positive_likelihood)]=0
        positive_likelihood = binary_fill_holes(positive_likelihood).astype(int)
        positive_likelihood = binary_dilation(positive_likelihood, iterations=2)
        positive_likelihood = binary_erosion(positive_likelihood, iterations=2)
        positive_likelihood = binary_fill_holes(positive_likelihood).astype(int)
        mask_refine[positive_likelihood>0]=nc
        
    return mask_refine, template


def print_training_log(epoch, epochs_total, epoch_loss, epoch_seg, model_dir):
    epoch_info = 'Epoch %d/%d' % (epoch, epochs_total)
    Loss_reg_ = 'Loss of Training Data: %.10f' % epoch_loss
    Dice_seg_ = 'Dice of Testing Data: %.10f' % epoch_seg
    
    l = ' - '.join((epoch_info, Loss_reg_, Dice_seg_))
    with open(model_dir + 'log.txt', 'a') as f:
        f.write("%s\n" % l)
    print(l)
    
def valid_range(sample_y):
    start = 0
    end = sample_y.shape[0]
    
    for i in range(sample_y.shape[0]):
        if (sample_y[i].sum())>0:
            start = i
            break
    
    for j in range(i, sample_y.shape[0]):
        if (sample_y[j].sum())==0:
            end = j
            break

    
    return start, end
   

def get_LargeSlice(sample_y):
    sum_=0
    for i in range(sample_y.shape[0]):
        if (sample_y[i].sum())>sum_:
            sum_ = sample_y[i].sum()
            idx = i
    return idx

def dice_volume(x, y):
    x, y = np.array(x), np.array(y)
    intersection = np.sum(x * y)
    union = np.sum(x) + np.sum(y)
    return 2 * intersection / union