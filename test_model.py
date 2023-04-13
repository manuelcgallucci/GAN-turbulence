import torch
import os
import numpy as np

import utility as ut
import nn_definitions as nn_d
from model_generator import CNNGeneratorBigConcat as CNNGenerator

model_names = ["mFwn3R","EDDVg4","1mnY3T","4K9Vke","6SgDhO","13whmJ","ORUKrz","k5cpTg","cyy81t","3fzNLY","XZ9Nph","2i5EIu","f5tOXl","VUwRRy","zXxm1p","VTsm0o","aKbLgh","7PU6YZ","Bnh0Pu"]

def test_model(scales, model_id=None, n_samples=64, len_=2**15, edge=4096, device="cuda"):
    data_dir = os.path.join('./generated', model_id)
    generator_dir = os.path.join(data_dir, "generator.pt")

    if not os.path.exists(generator_dir):
        print("Model {:s} doesnt exist!".format(model_id))
        return None 
    # Load the model 
    generator = CNNGenerator().to(device)
    generator.load_state_dict(torch.load(generator_dir))
    
    # Generate the samples
    noise = torch.randn((n_samples, 1, len_+2*edge), device=device)
    with torch.no_grad():
        generated_samples = generator(noise)
        generated_samples = generated_samples[:,:,edge:-edge]

    # Generate the structure functions from the data 

    struct = ut.calculate_structure(generated_samples, scales, device=device)
    struct_mean_generated = torch.mean(struct[:,:,:], dim=0)
    return struct_mean_generated
   

if __name__ == "__main__":
    print("Order: S2, Skewness, Flatness")
    
    device="cuda"

    nv=10
    uu=2**np.arange(0,13,1/nv)
    scales=np.unique(uu.astype(int))
    scales=scales[0:100]

    data_train = np.load('./data/data.npy')
    struct_mean_real = ut.calculate_structure(torch.Tensor(data_train[:,None,:]), scales, device=device)
    struct_mean_real = torch.mean(struct_mean_real[:,:,:], dim=0)
    
    for model_name in model_names:
        struct_mean_generated=test_model(scales, model_id=model_name, device=device)
        if not struct_mean_generated is None:
            mse_structure = torch.mean(torch.square(struct_mean_generated - struct_mean_real), dim=1)
            mse_structure = mse_structure.cpu().tolist()
            print(model_name, *mse_structure)
