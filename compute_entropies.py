import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

import utility as ut
import nn_definitions as nn_d
from model_generator import CNNGeneratorBigConcat as CNNGenerator

# All 52 models 
# model_names = ["mFwn3R", "EDDVg4", "1mnY3T", "4K9Vke", "6SgDhO", "13whmJ", "ORUKrz", "k5cpTg", "cyy81t", "3fzNLY", "XZ9Nph", "2i5EIu", "f5tOXl", "VUwRRy", "zXxm1p", "VTsm0o", "aKbLgh", "7PU6YZ", "Bnh0Pu", "gatNfU", "1qgc3k", "rsJbjN", "HLBoe3", "asIZKM", "02781t", "LFkfTr", "C4V4Lr", "81xhQr", "upbfBM", "xUzY79", "VMiMnc", "XcHcf4", "RwXQOQ", "i2i1vL", "1dT8m4", "tggFw3", "xnyGwo", "lvvMPG", "xupKrj", "etFaYm", "ZNEcCr", "guL8Zi", "rQ4xhZ", "o6jpvw", "H5Ewep", "GCJ2mB", "v5Cy7Z", "QDX49a", "o755la", "LPyYuW", "maCwVz", "w8bTmC"]
model_names = ["guL8Zi"]

def compute_entropy(s_real_samples, s_gen_samples,  divisions=[5,2350]):
	idx_1 = np.where(scales <= divisions[0])[-1][-1] + 1
	idx_2 = np.where(scales <= divisions[1])[-1][-1] + 1
	range_ = tuple([torch.min(s_real_samples[:,0,0]).item(), torch.max(s_real_samples[:,0,0]).item()])
	hist_real = torch.histogram(s_real_samples[:,0,0].detach().cpu(),bins=20, range=range_)
	hist_gen = torch.histogram(s_gen_samples[:,0,0].detach().cpu(),bins=20, range=range_)
	e = entropy(hist_real, hist_gen)
	print(e)
	# metrics_diss = torch.mean(torch.square(s_gen[:,:idx_1] - s_real[:,:idx_1]), dim=1)
	# metrics_diss = metrics_diss.cpu().tolist()
	# metrics_inert = torch.mean(torch.square(s_gen[:,idx_1:idx_2] - s_real[:,idx_1:idx_2]), dim=1)
	# metrics_inert = metrics_inert.cpu().tolist()
	# metrics_iteg = torch.mean(torch.square(s_gen[:,idx_2:] - s_real[:,idx_2:]), dim=1)
	# metrics_iteg = metrics_iteg.cpu().tolist()
	# metrics = [*metrics_diss, *metrics_inert, *metrics_iteg]
			
	return [1]
	 
	
def test_model(scales, model_id=None, n_samples=64, n_batches=4, len_=2**15, edge=4096, device="cuda", normalize=False):
	
	data_dir = os.path.join('./generated', model_id)
	generator_dir = os.path.join(data_dir, "generator.pt")

	if not os.path.exists(generator_dir):
		print("Model {:s} doesnt exist!".format(model_id))
		return None 
	# Load the model 
	generator = CNNGenerator().to(device)
	generator.load_state_dict(torch.load(generator_dir))

	# struct_means = torch.zeros((3, scales.shape[0]), device=device)

	all_structs = torch.zeros((n_batches*n_samples, 3, scales.shape[0]))
	for i_batch in range(n_batches):
		# Generate the samples
		noise = torch.randn((n_samples, 1, len_+2*edge), device=device)
		with torch.no_grad():
			generated_samples = generator(noise)
			generated_samples = generated_samples[:,:,edge:-edge]

		# Generate the structure functions from the data 
		struct = ut.calculate_structure(generated_samples, scales, device=device)
		all_structs[i_batch*n_samples: (1+i_batch)*n_samples] = struct.to("cpu")
		# struct_mean_generated = torch.mean(struct[:,:,:], dim=0)
		# struct_means += struct_mean_generated 
	return all_structs
   

if __name__ == "__main__":
	n_batches=4
	n_samples=8 #200
	len_=2**15
	
	normalize=True
	metrics_type="partial"
	
	if normalize:
		print("\nNormalized!")
	print("Metrics type:", metrics_type)
	print("Order: S2, Skewness, Flatness")

	device="cuda"

	nv=10
	uu=2**np.arange(0,13,1/nv)
	scales=np.unique(uu.astype(int))
	scales=scales[0:100]

	data_train = np.load('./data/data.npy')
	struct_real_samples = ut.calculate_structure(torch.Tensor(data_train[:,None,:]), scales, device=device)
	struct_mean_real = torch.mean(struct_real_samples[:,:,:], dim=0).to("cpu")
		
	for model_name in model_names:
		struct_gen_samples = test_model(scales, model_id=model_name, n_batches=n_batches, n_samples=n_samples, len_=len_, device=device, normalize=normalize)

		metrics = compute_entropy(struct_real_samples, struct_gen_samples)		
		print(model_name, *["{:.8f}".format(m).replace(".",",") for m in metrics])
