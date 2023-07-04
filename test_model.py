import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import re

import utility as ut
import nn_definitions as nn_d
from model_generator import CNNGeneratorBigConcat as CNNGenerator

# All 52 models 
# model_names = ["mFwn3R", "EDDVg4", "1mnY3T", "4K9Vke", "6SgDhO", "13whmJ", "ORUKrz", "k5cpTg", "cyy81t", "3fzNLY", "XZ9Nph", "2i5EIu", "f5tOXl", "VUwRRy", "zXxm1p", "VTsm0o", "aKbLgh", "7PU6YZ", "Bnh0Pu", "gatNfU", "1qgc3k", "rsJbjN", "HLBoe3", "asIZKM", "02781t", "LFkfTr", "C4V4Lr", "81xhQr", "upbfBM", "xUzY79", "VMiMnc", "XcHcf4", "RwXQOQ", "i2i1vL", "1dT8m4", "tggFw3", "xnyGwo", "lvvMPG", "xupKrj", "etFaYm", "ZNEcCr", "guL8Zi", "rQ4xhZ", "o6jpvw", "H5Ewep", "GCJ2mB", "v5Cy7Z", "QDX49a", "o755la", "LPyYuW", "maCwVz", "w8bTmC"]
# model_names = ["XO6y76", "DNza3x", "6gyLxg", "5UJL4D", "cHl4Dw", "6TBtpc", "w7zGFq", "maKGwp", "oLWtgC"]
# model_names = ["guL8Zi"]
# model_names = ["OvyWsl", "P7PmFA", "Q47i1m", "Ss0yK1"]
model_names = ["4CLqJd", "JQk4Aj"]

def plot_histogram(x, model_name, y=None, plt_path=None, name="histogram"):
	if plt_path is None:
		plt_path = os.path.join("./generated", model_name)
		if not os.path.exists(plt_path):
			return
			
	fig, ax = plt.subplots()
	ax.hist(x, alpha=0.5, label='Data 1')
	if y is not None:
		ax.hist(y, alpha=0.5, label='Data 2')
	ax.legend()
	plt.savefig(os.path.join(plt_path, name + ".png"))
	plt.close()

def normalize_struct(struct, type_="0-1"):
	# struct is of size 3,scales
	if type_ == "0-1":
		struct_min = struct.min(dim=1, keepdim=True)[0]
		struct_max = struct.max(dim=1, keepdim=True)[0]
		struct = (struct - struct_min) / (struct_max - struct_min)
	return struct

def compute_metrics(s_gen, s_real, scales, divisions = [5,2350], type_="full"):
	# S are the functions generated, of size 3xscales	
	if type_=="full":
		metrics = torch.mean(torch.square(s_gen - s_real), dim=1)
		metrics = metrics.cpu().tolist()
	elif type_=="partial":
		
		idx_1 = np.where(scales <= divisions[0])[-1][-1] + 1
		idx_2 = np.where(scales <= divisions[1])[-1][-1] + 1
		
		metrics_diss = torch.mean(torch.square(s_gen[:,:idx_1] - s_real[:,:idx_1]), dim=1)
		metrics_diss = metrics_diss.cpu().tolist()
			
		metrics_inert = torch.mean(torch.square(s_gen[:,idx_1:idx_2] - s_real[:,idx_1:idx_2]), dim=1)
		metrics_inert = metrics_inert.cpu().tolist()

		metrics_iteg = torch.mean(torch.square(s_gen[:,idx_2:] - s_real[:,idx_2:]), dim=1)
		metrics_iteg = metrics_iteg.cpu().tolist()
		
		metrics = [*metrics_diss, *metrics_inert, *metrics_iteg]
	elif type_ == "similarity":
		metrics = torch.nn.functional.cosine_similarity(s_gen, s_real, dim=1)
	elif type_ == "mahalanobis":
		# Calculate the inverse covariance matrix
		cov = torch.diag(torch.pow(torch.cat((a_std, b_std)), 2))
		inv_cov = torch.inverse(cov)

		# Calculate the difference vector between the two tensors
		diff = a - b

		# Calculate the Mahalanobis distance
		metrics = torch.sqrt(torch.matmul(torch.matmul(diff.T, inv_cov), diff))
	else:
		print("Metric not defined!")
		return None 
			
	return metrics 
	
def test_model(scales, data_dir, n_samples=64, n_batches=4, len_=2**15, edge=4096, device="cuda", normalize=False):
	
	generator_dir = os.path.join(data_dir, "generator.pt")
	if not os.path.exists(generator_dir):
		print("Path: {:s} doesnt exist!".format(generator_dir))
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
	
	# plot_histogram(all_structs[:,0,10].detach().cpu().numpy(), model_id, name="hist_gen")
	
	struct_means = torch.mean(all_structs, dim=0)
	if normalize:
		struct_means = normalize_struct(struct_means)
	return struct_means
   

if __name__ == "__main__":
	n_batches = 16 # 16
	n_samples = 200 # 200
	len_=2**15
	
	normalize=False
	metrics_type="similarity"
	
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
	struct_mean_real = ut.calculate_structure(torch.Tensor(data_train[:,None,:]), scales, device=device)
	
	# plot_histogram(struct_mean_real[:,0,10].detach().cpu().numpy(), model_names[0], name="hist_data")
	
	struct_mean_real = torch.mean(struct_mean_real[:,:,:], dim=0).to("cpu")
	if normalize:
		struct_mean_real = normalize_struct(struct_mean_real)
		
	for model_name in model_names:
		base_data_dir = os.path.join("./generated", "{:s}".format(model_name))
		
		
		struct_mean_generated=test_model(scales, data_dir=base_data_dir, n_batches=n_batches, n_samples=n_samples, len_=len_, device=device, normalize=normalize)
		if not struct_mean_generated is None:
			metrics = compute_metrics(struct_mean_generated, struct_mean_real, scales, type_=metrics_type)
			print(model_name, *["{:.8f}".format(m).replace(".",",") for m in metrics])

		for dir_ in os.listdir(base_data_dir):
			if "partial" in dir_:
				partial_id = int("".join(re.findall('\d+', dir_)))
				data_dir = os.path.join("./generated", "{:s}".format(model_name), dir_)
				struct_mean_generated=test_model(scales, data_dir=data_dir, n_batches=n_batches, n_samples=n_samples, len_=len_, device=device, normalize=normalize)
				if not struct_mean_generated is None:
					metrics = compute_metrics(struct_mean_generated, struct_mean_real, scales, type_=metrics_type)
					print("{:s}_p{:d}".format(model_name, partial_id), *["{:.8f}".format(m).replace(".",",") for m in metrics])
