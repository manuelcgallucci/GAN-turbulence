"""
Created on Fri Oct 28 21:12:45 2022

@author: Manuel
Added:
   - Regularized loss for the Generator
   - Corrected running metric calculations
   - Calculate and use s2 in loss_reg
   - Normalized both losses
"""
import torch 
from torch.utils.data import DataLoader
import numpy as np
from time import time
import os 
import sys

import dataloader as dl
import nn_definitions as nn_d
import utility as ut
# CNNGeneratorBCNocnn1
from model_generator import CNNGeneratorBigConcat as CNNGenerator
from model_discriminator import DiscriminatorMultiNet16_2 as Discriminator
#from model_discriminator import DiscriminatorSimpleCNN as Discriminator
import torch.multiprocessing as mp

# nohup python3 continue_training_no_structures.py > nohup_41.out &

def calculate_loss(criterion, predictions, target, weights, n_weights, device):
	loss = torch.zeros((1), device=device)
	for k in range(n_weights):
		loss += weights[k] * criterion(predictions[:,k], target)
	return loss

def combine_predictions(predictions, weights, n_weights, device):
	loss = torch.zeros((1), device=device)
	for k in range(n_weights):
		loss += weights[k] * predictions[:,k]
	return loss

def combine_losses_expAvg(loss_samples, loss_s2, loss_skewness, loss_flatness):
	alpha_samples = torch.exp(loss_samples).item()
	alpha_s2 = torch.exp(loss_s2).item()
	alpha_skewness = torch.exp(loss_skewness).item()
	alpha_flatness = torch.exp(loss_flatness).item()

	return  (alpha_samples * loss_samples + alpha_s2 * loss_s2 + alpha_skewness * loss_skewness + alpha_flatness * loss_flatness) / (alpha_samples + alpha_s2 + alpha_skewness + alpha_flatness)

def load_model_weights(model, weights_path, weights_name):
	
	if not weights_path is None:
		if os.path.exists(weights_path + weights_name):
			model.load_state_dict(torch.load(weights_path + weights_name))
		else:
			print("Model file: {:s} not found. Initializing with random weights".format(weights_path + weights_name))
			model = model.apply(nn_d.weights_init)
	else:
		model = model.apply(nn_d.weights_init)
	return model 
 
def normalize_struct(struct):
	# struct is of size 3,scales
	struct_min = struct.min(dim=1, keepdim=True)[0]
	struct_max = struct.max(dim=1, keepdim=True)[0]
	struct = (struct - struct_min) / (struct_max - struct_min)
	return struct
	     
def train_model_continue( continue_path, lr, epochs, batch_size, k_epochs_d, weights_sample_losses, weights_losses, data_type, data_stride, len_samples,out_dir, noise_size, measure_batch_size, save_threshold):

	n_weights = weights_sample_losses.size()[0]
	# print(n_weights)
	# Normalization for each loss

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Running on:", device)
	if device == "cpu":
		print("cuda device not found. Stopping the execution")
		return -1

	# Models and loading from checkpoint, else use random intialization
	generator = CNNGenerator().to(device)     
	generator = load_model_weights(generator, continue_path, '/generator.pt')

	discriminator = Discriminator().to(device)
	discriminator = load_model_weights(discriminator, continue_path, '/discriminator.pt')

	# criterion_MSE = torch.nn.MSELoss().to(device)
	optim_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
	optim_g = torch.optim.Adam(generator.parameters(),lr= lr, betas=(0.5, 0.999))

	# Train dataset 
	train_set, data_samples, data_len = dl.loadDataset(type=data_type, stride=data_stride, len_samples=len_samples)
	train_loader = DataLoader(train_set, batch_size=batch_size, num_workers = 0, shuffle = True, drop_last=False)

	if not continue_path is None:
		history = np.load(continue_path+"/metaEvo.npz")
		# Load already existing losses		
		loss_discriminator_array = torch.cat((torch.Tensor(history["loss_discriminator"]), torch.zeros((epochs))))

		loss_generator_array = torch.cat((torch.Tensor(history["loss_generator"]), torch.zeros((epochs))))

		measures_array = torch.cat((torch.Tensor(history["measures"]), torch.zeros((3, epochs))))
		epoch_offset = history["loss_discriminator"].shape[0]
	else:
		loss_discriminator_array = torch.zeros((epochs))

		loss_generator_array = torch.zeros((epochs))

		measures_array = torch.zeros((3,epochs))
		epoch_offset = 0

	## Optimizers
	optim_g.zero_grad()
	optim_d.zero_grad()

	# Pre-calculate the structure function for dataset
	# Use the average to compare 
	# Definition of the scales of analysis
	nv=10
	uu=2**np.arange(0,13,1/nv)
	scales=np.unique(uu.astype(int))
	scales=scales[0:100]

	# Calculate on the cpu and then put in GPU at training time 
	# These take some Mb of GPUram otherwise no time difference
	saved_partial_res = 1
	data_structure_functions = []
	struct_mean_real = torch.zeros((3,scales.shape[0]),device="cpu")
	for _, data_ in enumerate(train_loader):
		data_ = data_.to("cpu").float()
		structure_f = ut.calculate_structure_noInplace(data_, scales, device="cpu")        
		struct_mean_real = struct_mean_real + torch.mean(structure_f, dim=0)
		data_structure_functions.append(structure_f)

	struct_mean_real = struct_mean_real.to(device)
	struct_mean_real = normalize_struct(struct_mean_real)
	start_time = time()
	for epoch in range(epochs):

		for batch_idx, data_ in enumerate(train_loader):
			# start_time_batch = time()

			data_ = data_.to(device).float()
			# data_ = torch.unsqueeze(data_, dim=1)
			batch_size_ = data_.shape[0]

			## TRAIN DISCRIMINATOR
			for kd in range(k_epochs_d):
				discriminator.zero_grad()

				# optim_d.zero_grad()
				## True samples				
				predictions_real = discriminator(data_)
				predictions_real = combine_predictions(predictions_real, weights_sample_losses, n_weights, device)

				## False samples (Create random noise and run the generator on them)
				noise = torch.randn((batch_size_, noise_size[0], noise_size[1]), device=device)
				with torch.no_grad():
					fake_samples = generator(noise)
									
				# Fake samples
				predictions_fake = discriminator(fake_samples)
				predictions_fake = combine_predictions(predictions_fake, weights_sample_losses, n_weights, device)

				# Combine the losses 
				loss_discriminator = torch.mean(predictions_fake) - torch.mean(predictions_real)
				
				loss_discriminator.backward()

				# Discriminator optimizer step
				optim_d.step()

				loss_discriminator_array[epoch+epoch_offset] += loss_discriminator.item() / k_epochs_d

			## TRAIN GENERATOR
			generator.zero_grad()

			noise = torch.randn((batch_size_, noise_size[0], noise_size[1]), device=device)
			generated_signal = generator(noise) 

			# Fake samples
			predictions = discriminator(generated_signal)
			predictions = combine_predictions(predictions, weights_sample_losses, n_weights, device)
			
			loss_generator = -torch.mean(predictions)
			
			loss_generator.backward()
			optim_g.step()

			loss_generator_array[epoch+epoch_offset] += loss_generator.item()
		print('Epoch [{}/{}] -\t Generator Loss: {:7.4f} \t/\t\t Discriminator Loss: {:7.4f}'.format(epoch+1, epochs, loss_generator_array[epoch+epoch_offset], loss_discriminator_array[epoch+epoch_offset]))
		sys.stdout.flush()
		
		# If the mean S2, Skewness and Flatness are 'good enough' then save the model
		struct_mean_generated = torch.zeros((3, scales.shape[0]), device=device)
		with torch.no_grad():
			for i_batch in range(measure_batch_size):
				# Generate the samples
				noise = torch.randn((batch_size_, noise_size[0], noise_size[1]), device=device)		
				generated_signal = generator(noise) 
				
				# Generate the structure functions from the data 
				struct = ut.calculate_structure_noInplace(generated_signal, scales, device=device)
				struct_mean_generated += torch.mean(struct[:,:,:], dim=0)
		
		struct_mean_generated = normalize_struct(struct_mean_generated / measure_batch_size)
		
		# measures = torch.mean( torch.square(struct_mean_generated - struct_mean_real), dim=1)
		measures = torch.nn.functional.cosine_similarity(struct_mean_generated, struct_mean_real, dim=1)
		measures_array[:,epoch+epoch_offset] = measures
		
		if torch.mean(measures).item() >= save_threshold:
			base_partial_dir = os.path.join(out_dir, "partial_{:d}".format(saved_partial_res))
			os.mkdir(base_partial_dir)
			torch.save(generator.state_dict(), os.path.join(base_partial_dir, 'generator.pt'))
			torch.save(discriminator.state_dict(), os.path.join(base_partial_dir, 'discriminator.pt'))
						
			with open( os.path.join(base_partial_dir, "partial_meta.txt"), "w") as f:
				f.write("Partial epoch save: {:d} ({:d} in training)".format(epoch+epoch_offset, epoch) + '\n')
				f.write("S2 MSE: " + str(measures[0].item()) + '\n')
				f.write("Skewness MSE: " + str(measures[1].item()) + '\n')
				f.write("Flatness MSE: " + str(measures[2].item()) + '\n')
			saved_partial_res = saved_partial_res + 1
			
			save_threshold = torch.mean(measures).item()

	end_time = time()

	torch.save(generator.state_dict(), out_dir + '/generator.pt')
	torch.save(discriminator.state_dict(), out_dir + '/discriminator.pt')
	
	# TODO the cpu detach is depr and could be replaced by numpy()
	np.savez(out_dir+"/metaEvo.npz", \
			loss_discriminator = loss_discriminator_array.numpy(), \
			loss_generator = loss_generator_array.numpy(), \
			
			measures = measures_array.numpy())

	
	with open( os.path.join(out_dir, "time.txt"), "w") as f:
		f.write("Total time to train in seconds: {:f}".format(end_time - start_time))

	return out_dir

# nohup python3 continue_training.py > nohup_1.out &
if __name__ == '__main__':                        
	data_type = "full" # full samples from the original data     
	data_stride = 2**15
	len_samples = 2**15
	noise_size=(1, len_samples)

	lr = 0.001
	epochs = 500
	batch_size = 32
	measure_batch_size = 8
	
	k_epochs_d = 2

	#weights_sample_losses = torch.Tensor([1,1,0.5,0.5,0.5,0.5,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125 ])
	weights_sample_losses = torch.Tensor([1])
	weights_losses = [1.0, 0.0, 0.0, 0.0] # weights for sample, s2, skewness, flatness
	continue_training = None
	
	save_threshold = 0.5
	out_dir = './generated'
	out_dir = ut.get_dir(out_dir)

	meta_dict = {
		"lr":lr,
		"epochs":epochs,
		"batch_size":batch_size,
		"k_epochs_d":k_epochs_d,
		"out_dir":out_dir,
		"weights_sample_losses":weights_sample_losses,
		"weights_losses":weights_losses,
		"data_type_loading":data_type,
		"data_type_stride":data_stride,
		"len_samples":len_samples,
		"continue_training":continue_training,
		"measure_batch_size":measure_batch_size,
		"train_file_type": "Training done for a combined discriminator loss without structure functions. Simple CNN",
	}

	ut.save_meta(meta_dict, out_dir)
	print("Begun training " + out_dir + " " + str(weights_losses))
	sys.stdout.flush()
	if not continue_training is None:
		continue_path = os.path.join("./generated",continue_training)
	else:
		continue_path=None
	train_model_continue(continue_path, lr, epochs, batch_size, k_epochs_d, weights_sample_losses, weights_losses, data_type, data_stride, len_samples, out_dir, noise_size, measure_batch_size, save_threshold)

	print("Finished training " + out_dir + " " + str(weights_losses))
