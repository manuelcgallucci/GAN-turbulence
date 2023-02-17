
import numpy as np
import torch
import random, string
import os 

# Tested at 50 tests, 32 batch size, 2**15 length, 100 scales
# 0.017 gpu
def calculate_s2(signal, scales, device="cpu"):
    '''
    signal is the signal of study and scales is an array with the values of the scales of analysis
    '''      
    s2 = torch.zeros((signal.shape[0],1,len(scales)), dtype=torch.float32, device=device)

    # We normalize the image by centering and standarizing it
    Nreal=signal.size()[0]
    tmp = torch.zeros(signal.shape, device=device)    
    for ir in range(Nreal):
        nanstdtmp = torch.sqrt(torch.nanmean(torch.abs(signal[ir]-torch.nanmean(signal[ir]))**2))
        tmp[ir,0,:] = (signal[ir]-torch.nanmean(signal[ir]))/nanstdtmp   

    for idx, scale in enumerate(scales):
        s2[:,:,idx] = torch.log(torch.mean(torch.square(tmp[:,:,scale:]-tmp[:,:,:-scale]), dim=2))
        
    return s2



def get_dir(dir, length=6):
    name = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    path_ = os.path.join(dir, name)
    while os.path.isfile(path_):
        name = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        path_ = os.path.join(dir, name)
    os.mkdir(path_)
    return path_

def save_meta(meta_dict, meta_dir, meta_name="meta.txt"):
    with open( os.path.join(meta_dir, meta_name), "w") as f:
        for k, v in meta_dict.items():
            f.write("{:s}: ".format(k) + str(v) + '\n')

        f.write("\nreg: \n")