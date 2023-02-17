from torch.utils.data import Dataset
import numpy as np
# Define DataLoader
class DatasetCustom(Dataset):
    def __init__(self, data):
        # Data should be an array 
        self.data = data
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return np.expand_dims( self.data[index,:], axis=0)
        #return self.data[index,:]