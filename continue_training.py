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
from model_discriminator import DiscriminatorMultiNet16_4 as Discriminator
from model_discriminator import DiscriminatorStructures as DiscriminatorStructures
import torch.multiprocessing as mp

def calculate_loss(criterion, predictions, target, weights, n_weights, device):
    loss = torch.zeros((1), device=device)
    for k in range(n_weights):
        loss += weights[k] * criterion(predictions[:,k], target)
    return loss

def combine_losses_expAvg(loss_samples, loss_s2, loss_skewness, loss_flatness):
	alpha_samples = torch.exp(loss_samples).item()
	alpha_s2 = torch.exp(loss_s2).item()
	alpha_skewness = torch.exp(loss_skewness).item()
	alpha_flatness = torch.exp(loss_flatness).item()
	
	return  (alpha_samples * loss_samples + alpha_s2 * loss_s2 + alpha_skewness * loss_skewness + alpha_flatness * loss_flatness) / (alpha_samples + alpha_s2 + alpha_skewness + alpha_flatness)

def load_model_weights(model, weights_path):
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
    else:
        print("Model file: {:s} not found. Initializing with random weights".format(weights_path))
        model = model.apply(nn_d.weights_init)
    return model 
         
def train_model_continue( lr, epochs, batch_size, k_epochs_d, weights_sample_losses, weights_losses, data_type, data_stride, len_samples,out_dir, noise_size):
    
    n_weights = weights_sample_losses.size()[0]
    # Normalization for each loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)
    if device == "cpu":
        print("cuda device not found. Stopping the execution")
        return -1

    # Models and loading from checkpoint, else use random intialization
    generator = CNNGenerator().to(device)     
    generator = load_model_weights(generator, out_dir + '/generator.pt')

    discriminator = Discriminator().to(device)
    discriminator = load_model_weights(discriminator, out_dir + '/discriminator.pt')

    discriminator_s2 = DiscriminatorStructures().to(device)
    discriminator_s2 = load_model_weights(discriminator_s2, out_dir + '/discriminator_s2.pt')

    discriminator_skewness = DiscriminatorStructures().to(device)
    discriminator_skewness = load_model_weights(discriminator_skewness, out_dir + '/discriminator_skewness.pt')

    discriminator_flatness = DiscriminatorStructures().to(device)
    discriminator_flatness = load_model_weights(discriminator_flatness, out_dir + '/discriminator_flatness.pt')
    
    # define loss and optimizers
    criterion_BCE = torch.nn.BCELoss().to(device)
    # criterion_MSE = torch.nn.MSELoss().to(device)
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    optim_ds2 = torch.optim.Adam(discriminator_s2.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_dskewness = torch.optim.Adam(discriminator_skewness.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_dflatness = torch.optim.Adam(discriminator_flatness.parameters(), lr=lr, betas=(0.5, 0.999))

    optim_g = torch.optim.Adam(generator.parameters(),lr= lr, betas=(0.5, 0.999))

    # Train dataset 
    train_set, data_samples, data_len = dl.loadDataset(type=data_type, stride=data_stride, len_samples=len_samples)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers = 0, shuffle = True, drop_last=False)
    
    history = np.load(out_dir+"/metaEvo.npz")

    # Load already existing losses
    loss_real_array = torch.cat((history["loss_real"], torch.zeros((epochs)))) 
    loss_real_s2_array = torch.cat((history["loss_real_s2"], torch.zeros((epochs))))
    loss_real_skewness_array = torch.cat((history["loss_real_skewness"], torch.zeros((epochs))))
    loss_real_flatness_array = torch.cat((history["loss_real_flatness"], torch.zeros((epochs))))
    loss_real_total_array = torch.cat((history["loss_real_total"], torch.zeros((epochs))))

    loss_fake_array = torch.cat((history["loss_fake"], torch.zeros((epochs))))
    loss_fake_s2_array = torch.cat((history["loss_fake_s2"], torch.zeros((epochs))))
    loss_fake_skewness_array = torch.cat((history["loss_fake_skewness"], torch.zeros((epochs))))
    loss_fake_flatness_array = torch.cat((history["loss_fake_flatness"], torch.zeros((epochs))))
    loss_fake_total_array = torch.cat((history["loss_fake_total"], torch.zeros((epochs))))

    loss_discriminator_array = torch.cat((history["loss_discriminator"], torch.zeros((epochs))))

    loss_g_array = torch.cat((history["loss_g"], torch.zeros((epochs))))
    loss_g_s2_array = torch.cat((history["loss_g_s2"], torch.zeros((epochs))))
    loss_g_skewness_array = torch.cat((history["loss_g_skewness"], torch.zeros((epochs))))
    loss_g_flatness_array = torch.cat((history["loss_g_flatness"], torch.zeros((epochs))))

    loss_generator_array = torch.cat((history["loss_generator"], torch.zeros((epochs))))

    epoch_offset = history["loss_discriminator"].shape[0]
    
    ## Optimizers
    optim_g.zero_grad()
    optim_d.zero_grad()
    optim_ds2.zero_grad()
    optim_dskewness.zero_grad()
    optim_dflatness.zero_grad()

    # Take the target ones and zeros for the batch size and for the last (not complete batch)
    target_ones_full = torch.ones((batch_size), device=device)
    target_ones_partial = torch.ones((data_samples - batch_size * int(data_samples / batch_size)), device=device)
    target_ones = [target_ones_full, target_ones_partial]

    target_zeros_full = torch.zeros((batch_size), device=device)
    target_zeros_partial = torch.ones((data_samples - batch_size * int(data_samples / batch_size)), device=device)
    target_zeros = [target_zeros_full, target_zeros_partial]

    last_batch_idx = np.ceil(data_samples / batch_size) - 1

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
    struct_mean_real = torch.Tensor(device="cpu")
    for _, data_ in enumerate(train_loader):
        data_ = data_.to("cpu").float()
        structure_f = ut.calculate_structure_noInplace(data_, scales, device="cpu")        
        struct_mean_real = struct_mean_real + torch.mean(structure_f, dim=0)
        data_structure_functions.append(structure_f)

    struct_mean_real = struct_mean_real.to(device)
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
                
                discriminator_s2.zero_grad()
                discriminator_skewness.zero_grad()
                discriminator_flatness.zero_grad()

                # optim_d.zero_grad()
                ## True samples
                
                predictions = discriminator(data_)
                loss_real = calculate_loss(criterion_BCE, predictions, target_ones[int(batch_idx == last_batch_idx)], weights_sample_losses, n_weights, device)
                
                structure_f = data_structure_functions[batch_idx].to(device)
                # structure_f = data_structure_functions[batch_idx]
                
                loss_real_s2 = criterion_BCE(discriminator_s2(structure_f[:,0,:])[:,0], target_ones[int(batch_idx == last_batch_idx)])
                loss_real_skewness = criterion_BCE(discriminator_skewness(structure_f[:,1,:])[:,0], target_ones[int(batch_idx == last_batch_idx)])
                loss_real_flatness = criterion_BCE(discriminator_flatness(structure_f[:,2,:])[:,0], target_ones[int(batch_idx == last_batch_idx)])
                
                loss_real_total = weights_losses[0] * loss_real + weights_losses[1] * loss_real_s2 + weights_losses[2] * loss_real_skewness + weights_losses[3] * loss_real_flatness
                # loss_real_total = combine_losses_expAvg( loss_real, loss_real_s2, loss_real_skewness, loss_real_flatness)
                
                ## False samples (Create random noise and run the generator on them)
                noise = torch.randn((batch_size_, noise_size[0], noise_size[1]), device=device)
                with torch.no_grad():
                    fake_samples = generator(noise)
                                    
                # Fake samples
                predictions = discriminator(fake_samples)
                loss_fake = calculate_loss(criterion_BCE, predictions, target_zeros[int(batch_idx == last_batch_idx)], weights_sample_losses, n_weights, device)
                
                structure_f = ut.calculate_structure_noInplace(fake_samples, scales, device=device)

                loss_fake_s2 = criterion_BCE(discriminator_s2(structure_f[:,0,:])[:,0], target_zeros[int(batch_idx == last_batch_idx)])
                loss_fake_skewness = criterion_BCE(discriminator_skewness(structure_f[:,1,:])[:,0], target_zeros[int(batch_idx == last_batch_idx)])
                loss_fake_flatness = criterion_BCE(discriminator_flatness(structure_f[:,2,:])[:,0], target_zeros[int(batch_idx == last_batch_idx)])

                
                loss_fake_total = weights_losses[0] * loss_fake + weights_losses[1] * loss_fake_s2 + weights_losses[2] * loss_fake_skewness + weights_losses[3] * loss_fake_flatness
                # loss_fake_total = combine_losses_expAvg( loss_fake, loss_fake_s2, loss_fake_skewness, loss_fake_flatness)
                
                # Combine the losses 
                loss_discriminator = (loss_real_total + loss_fake_total) / 2
                
                # loss_real.backward()
                # loss_fake.backward()
                loss_discriminator.backward()

                # Discriminator optimizer step
                optim_d.step()
                optim_ds2.step()
                optim_dskewness.step()
                optim_dflatness.step()

                loss_real_array[epoch+epoch_offset] += loss_real.item() / k_epochs_d
                loss_real_s2_array[epoch+epoch_offset] += loss_real_s2.item() / k_epochs_d
                loss_real_skewness_array[epoch+epoch_offset] += loss_real_skewness.item() / k_epochs_d
                loss_real_flatness_array[epoch+epoch_offset] += loss_real_flatness.item() / k_epochs_d
                loss_real_total_array[epoch+epoch_offset] += loss_real_total.item() / k_epochs_d

                loss_fake_array[epoch+epoch_offset] += loss_fake.item() / k_epochs_d
                loss_fake_s2_array[epoch+epoch_offset] += loss_fake_s2.item() / k_epochs_d
                loss_fake_skewness_array[epoch+epoch_offset] += loss_fake_skewness.item() / k_epochs_d
                loss_fake_flatness_array[epoch+epoch_offset] += loss_fake_flatness.item() / k_epochs_d
                loss_fake_total_array[epoch+epoch_offset] += loss_fake_total.item() / k_epochs_d

                loss_discriminator_array[epoch+epoch_offset] += loss_discriminator.item() / k_epochs_d

                ## TRAIN GENERATOR
                generator.zero_grad()

                noise = torch.randn((batch_size_, noise_size[0], noise_size[1]), device=device)
                generated_signal = generator(noise) 

                # Fake samples
                predictions = discriminator(generated_signal)
                loss_g_samples = calculate_loss(criterion_BCE, predictions, target_ones[int(batch_idx == last_batch_idx)], weights_sample_losses, n_weights, device)

                structure_f = ut.calculate_structure_noInplace(generated_signal, scales, device=device)

                loss_g_s2 = criterion_BCE(discriminator_s2(structure_f[:,0,:])[:,0], target_ones[int(batch_idx == last_batch_idx)])
                loss_g_skewness = criterion_BCE(discriminator_skewness(structure_f[:,1,:])[:,0], target_ones[int(batch_idx == last_batch_idx)])
                loss_g_flatness = criterion_BCE(discriminator_flatness(structure_f[:,2,:])[:,0], target_ones[int(batch_idx == last_batch_idx)])
                
                loss_generator = weights_losses[0] * loss_g_samples + weights_losses[1] * loss_g_s2 + weights_losses[2] * loss_g_skewness + weights_losses[3] * loss_g_flatness
                
                loss_generator.backward()
                optim_g.step()

                loss_g_array[epoch+epoch_offset] += loss_g_samples.item()
                loss_g_s2_array[epoch+epoch_offset] += loss_g_s2.item()
                loss_g_skewness_array[epoch+epoch_offset] += loss_g_skewness.item()
                loss_g_flatness_array[epoch+epoch_offset] += loss_g_flatness.item()

                loss_generator_array[epoch+epoch_offset] += loss_generator.item()

                # If the mean S2, Skewness and Flatness are 'good enough' then save the model
                measures = torch.mean(torch.square(torch.mean(structure_f[:,:,:], dim=0) - struct_mean_real), dim=1)
                print(measures)
                if (measures < 0.0001).all():
                    base_partial_dir = os.path.join(out_dir, "partial_{:d}".format(saved_partial_res))
                    torch.save(generator.state_dict(), os.path.join(base_partial_dir, 'generator.pt'))
                    with open( os.path.join(base_partial_dir, "partial_meta.txt"), "w") as f:
                        f.write("Partial epoch save: {:d}".format(epoch+epoch_offset) + '\n')
                        f.write("S2 MSE: {:f}".format() + '\n')
                        f.write("Skewness MSE: {:f}".format() + '\n')
                        f.write("Flatness MSE: {:f}".format() + '\n')
                    saved_partial_res = saved_partial_res + 1
        
        # print('Epoch [{}/{}] -\t Generator Loss: {:7.4f} \t/\t\t Discriminator Loss: {:7.4f}'.format(epoch+1, epochs, loss_generator_array[epoch+epoch_offset], loss_discriminator_array[epoch+epoch_offset]))
        # sys.stdout.flush()

    end_time = time()
    
    np.savez(out_dir+"/metaEvo.npz", \
            loss_fake = loss_fake_array.cpu().detach().numpy(), \
            loss_fake_s2 = loss_fake_s2_array.cpu().detach().numpy(), \
            loss_fake_skewness = loss_fake_skewness_array.cpu().detach().numpy(), \
            loss_fake_flatness = loss_fake_flatness_array.cpu().detach().numpy(), \
            loss_fake_total = loss_fake_total_array.cpu().detach().numpy(), \
            
            loss_real = loss_real_array.cpu().detach().numpy(), \
            loss_real_s2 = loss_real_s2_array.cpu().detach().numpy(), \
            loss_real_skewness = loss_real_skewness_array.cpu().detach().numpy(), \
            loss_real_flatness = loss_real_flatness_array.cpu().detach().numpy(), \
            loss_real_total = loss_real_total_array.cpu().detach().numpy(), \
            
            loss_g = loss_g_array.cpu().detach().numpy(), \
            loss_g_s2 = loss_g_s2_array.cpu().detach().numpy(), \
            loss_g_skewness = loss_g_skewness_array.cpu().detach().numpy(), \
            loss_g_flatness = loss_g_flatness_array.cpu().detach().numpy(), \
            
            loss_discriminator = loss_discriminator_array.cpu().detach().numpy(), \
            loss_generator = loss_generator_array.cpu().detach().numpy())

    torch.save(generator.state_dict(), out_dir + '/generator.pt')
    torch.save(discriminator.state_dict(), out_dir + '/discriminator.pt')
    torch.save(discriminator_s2.state_dict(), out_dir + '/discriminator_s2.pt')
    torch.save(discriminator_skewness.state_dict(), out_dir + '/discriminator_skewness.pt')
    torch.save(discriminator_flatness.state_dict(), out_dir + '/discriminator_flatness.pt')

    with open( os.path.join(out_dir, "time.txt"), "w") as f:
        f.write("Total time to train in seconds: {:f}".format(end_time - start_time))

    return out_dir

def test_print(str_):
	print(str_)



# nohup python3 train_model_hyperparams.py > nohup_1.out &
if __name__ == '__main__':                        
    data_type = "full" # full samples from the original data     
    data_stride = 2**15
    len_samples = 2**15
    noise_size=(1, len_samples)

    lr = 0.002
    epochs = 250
    batch_size = 16
    k_epochs_d = 2

    weights_sample_losses = torch.Tensor([1,1,0.5,0.5,0.5,0.5,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25])
    weights_losses = [0.5, 0.2, 0.15, 0.15]
    continue_training = 'guL8Zi'

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
        "train_file_type": "Training done for a combined discriminator loss with structure functions",
    }

    ut.save_meta(meta_dict, out_dir)
    print("Begun training " + out_dir + " " + str(weights_losses))
    sys.stdout.flush()
    train_model_continue(continue_training, lr, epochs, batch_size, k_epochs_d, weights_sample_losses, weights_losses, data_type, data_stride, len_samples, out_dir, noise_size)

    print("Finished training " + out_dir + " " + str(weights_losses))
