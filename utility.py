
import numpy as np
import torch

# Tested at 50 tests, 32 batch size, 2**15 length, 100 scales
# 0.916 / 1.01 / 1.10 avg on gpu
def calculate_s2(signal, scales, device="cpu"):
    '''
    signal is the signal of study and scales is an array with the values of the scales of analysis
    '''  
    
    Nreal=signal.size()[0]
    Struc=torch.zeros((Nreal,1,len(scales)), dtype=torch.float32, device=device)
        
    for ir in range(Nreal):
        
        # We normalize the image by centering and standarizing it
        nanstdtmp=torch.sqrt(torch.nanmean(torch.abs(signal[ir]-torch.nanmean(signal[ir]))**2))
        tmp=(signal[ir]-torch.nanmean(signal[ir]))/nanstdtmp   

        for isc in range(len(scales)):
            scale=int(scales[isc])
                
            incrs=tmp[0,scale:]-tmp[0,:-scale]
            incrs=incrs[~torch.isnan(incrs)]
            Struc[ir,0,isc]=torch.log(torch.nanmean(incrs.flatten()**2))
        
    return Struc

# Tested at 50 tests, 32 batch size, 2**15 length, 100 scales
# 0.017 gpu
def calculate_s2_v2(signal, scales, device="cpu"):
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
