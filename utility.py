
import numpy as np
import torch

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
