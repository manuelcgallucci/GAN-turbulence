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

# DiscriminatorMultiNetWeightedAvg DiscriminatorMultiNetNo512 DiscriminatorMultiNet16_4

plot_metadata_training = True

# nohup python3 train_model_step_discriminator.py > nohup_1.out &

def calculate_loss(criterion, predictions, target, weights, n_weights, device):
    loss = torch.zeros((1), device=device)
    for k in range(n_weights):
        loss += weights[k] * criterion(predictions[:,k], target)

    # Make the predictions 
    mean_prediction = torch.mean(predictions,dim=1)
    return loss, mean_prediction

def train_model( lr, epochs, batch_size, k_epochs_d, k_epochs_g, alpha, beta, gamma, weigths, data_type, data_stride, len_samples,out_dir, noise_size=(1,2**15)):
    
    n_weights = weights.size()[0]
    # alpha serves as the parameter in the generator regularization loss
    alpha_comp = 1 - alpha
    # beta serves as the multiplier for the total generator loss
    # gamma serves as the multiplier for the total disctiminator loss
    # Normalization for each loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)
    if device == "cpu":
        if input('Running on cpu, continue? [y/n]') == 'y':
            return 1
        
    # Models with Weight initialization
    generator = CNNGenerator().to(device)
    generator = generator.apply(nn_d.weights_init)

    discriminator = Discriminator().to(device)
    discriminator = discriminator.apply(nn_d.weights_init)

    # define loss and optimizers
    criterion_BCE = torch.nn.BCELoss().to(device)
    criterion_MSE = torch.nn.MSELoss().to(device)
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_g = torch.optim.Adam(generator.parameters(),lr= lr, betas=(0.5, 0.999))

    # Train dataset 
    # data_train = torch.Tensor(np.load('./data/data.npy')) # Nsamples x L (L: length)
    # train_set = dl.DatasetCustom(data_train)
    # data_samples = data_train.size()[0]
    # data_len = data_train.size()[1]

    train_set, data_samples, data_len = dl.loadDataset(type=data_type, stride=data_stride, len_samples=len_samples)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers = 0, shuffle = True, drop_last=False)

    loss_d_fake_array = torch.zeros((epochs))
    loss_d_real_array = torch.zeros((epochs))
    loss_d_array = torch.zeros((epochs))
    acc_d_fake_array = torch.zeros((epochs)) # Accuracy or discriminator on the fake samples
    acc_d_real_array = torch.zeros((epochs))

    loss_g_array = torch.zeros((epochs))
    loss_reg_array = torch.zeros((epochs))
    loss_gen_array = torch.zeros((epochs))

    optim_g.zero_grad()
    optim_d.zero_grad()

    # Pre-calculate the structure function for dataset
    # Use the average to compare 
    # Definition of the scales of analysis
    nv=10
    uu=2**np.arange(0,13,1/nv)
    scales=np.unique(uu.astype(int))
    scales=scales[0:100]
    
    # s2 = ut.calculate_s2(torch.cumsum(data_train[:,None,:], dim=2), scales, device=device)
    # s2 = ut.calculate_s2(data_train[:,None,:], scales, device=device)
    # mean_s2 = torch.mean(s2, dim=[0,1]) # This gives the s2 tensor of size (len(scales))
    # mean_s2 = mean_s2[None, None, :] 

    # Take the target ones and zeros for the batch size and for the last (not complete batch)

    target_ones_full = torch.ones((batch_size), device=device)
    target_ones_partial = torch.ones((data_samples - batch_size * int(data_samples / batch_size)), device=device)
    target_ones = [target_ones_full, target_ones_partial]

    target_zeros_full = torch.zeros((batch_size), device=device)
    target_zeros_partial = torch.ones((data_samples - batch_size * int(data_samples / batch_size)), device=device)
    target_zeros = [target_zeros_full, target_zeros_partial]
    
    last_batch_idx = np.ceil(data_samples / batch_size) - 1
    
    start_time = time()
    for epoch in range(epochs):

        for batch_idx, data_ in enumerate(train_loader):

            data_ = data_.to(device).float()
            # data_ = torch.unsqueeze(data_, dim=1)
            batch_size_ = data_.shape[0]

            ## TRAIN DISCRIMINATOR
            for kd in range(k_epochs_d):
                discriminator.zero_grad()
                
                # optim_d.zero_grad()
                ## True samples
                # p16384_0, p16384_1, p8192_0, p8192_1, p8192_2, p8192_3, p4096_0, p4096_1, p4096_2, p4096_3, p4096_4, p4096_5, p4096_6, p4096_7 = discriminator(data_)
                # p16384_0, p16384_1 = discriminator(data_)
                predictions = discriminator(data_)
                loss_real, mean_prediction_real = calculate_loss(criterion_BCE, predictions, target_ones[int(batch_idx == last_batch_idx)], weights, n_weights, device)
                
                # loss_real = 1.0 * (l16384_0_real + l16384_1_real)
                # concatenated_predictions = torch.cat((p16384_0, p16384_1),dim=1)
                # mean_prediction_real = torch.mean(concatenated_predictions,dim=1)

                ## False samples (Create random noise and run the generator on them)
                noise = torch.randn((batch_size_, noise_size[0], noise_size[1]), device=device)
                with torch.no_grad():
                    fake_samples = generator(noise)
                                    
                # Fake samples
                predictions = discriminator(fake_samples)
                loss_fake, mean_prediction_fake = calculate_loss(criterion_BCE, predictions, target_zeros[int(batch_idx == last_batch_idx)], weights, n_weights, device)
                
                # Combine the losses 
                loss_d = gamma * (loss_real + loss_fake) / 2
                
                # loss_real.backward()
                # loss_fake.backward()
                loss_d.backward()

                # Discriminator optimizer step
                optim_d.step()
                
                loss_d_real_array[epoch] += loss_real.item() / k_epochs_d
                loss_d_fake_array[epoch] += loss_fake.item() / k_epochs_d
                loss_d_array[epoch] += loss_d.item() / k_epochs_d

            # print("Fake mean predictions:", mean_prediction_fake)
            # print("Real mean predictions:", mean_prediction_real)
            # print("\tPredictions:", p16384_0, "\n\t",p16384_1)
            # If pred_fake is all zeros then acc should be 1.0
            # We want this to be around 0.5. 1.0 means perfect accuracy (the generated samples are not similar to the samples)
            acc_d_fake_array[epoch] += torch.sum(mean_prediction_fake < 0.5).item()
            acc_d_real_array[epoch] += torch.sum(mean_prediction_real >= 0.5).item()
            
            for kg in range(k_epochs_g):
                    
                #optim_g.zero_grad()
                ## TRAIN GENERATOR
                generator.zero_grad()

                noise = torch.randn((batch_size_, noise_size[0], noise_size[1]), device=device)
                generated_signal = generator(noise) 
                # Cut samples (no)

                # Fake samples
                predictions = discriminator(generated_signal)
                loss_g, mean_predictions_g = calculate_loss(criterion_BCE, predictions, target_ones[int(batch_idx == last_batch_idx)], weights, n_weights, device)
                
                # E [( X * cumsum(Z) ) ^2]
                # loss_reg = torch.mean(torch.square(torch.mul(noise,torch.cumsum(generated_signal, dim=2))))
                # E[X * cumusm(z)]
                # loss_reg = torch.mean(torch.mul(noise,torch.cumsum(generated_signal, dim=2)))
                # E[Z]^2
                # loss_reg = torch.square(torch.mean(generated_signal) * batch_size_)
                
                # Structure function loss
                # Broadcast s2 into all structure function estimations
                # generated_s2 = ut.calculate_s2(torch.cumsum(generated_signal,dim=2), scales, device=device)
                # loss_reg = criterion_MSE(generated_s2, mean_s2)
                
                # loss_gen = beta * ( alpha_comp * loss_g + alpha *  loss_reg )

                loss_gen = beta * loss_g

                loss_gen.backward()
                optim_g.step()

                loss_gen_array[epoch] += loss_gen.item() / k_epochs_g
                # loss_reg_array[epoch] += loss_reg.item()
                loss_g_array[epoch] += loss_g.item() / k_epochs_g


        # THESE HAVE TO BE DEVIDED BY THE NUMBER OF BATCHES
        acc_d_fake_array[epoch] = acc_d_fake_array[epoch] / data_samples
        acc_d_real_array[epoch] = acc_d_real_array[epoch] / data_samples

        if epoch%5 == 0:
            print('Epoch [{}/{}] -\t Generator Loss: {:7.4f} \t/\t\t Discriminator Loss: {:7.4f} || acc(fake): {:7.4f} , acc(real): {:7.4f} ||'.format(epoch+1, epochs, loss_gen_array[epoch], loss_d_array[epoch], acc_d_fake_array[epoch], acc_d_real_array[epoch]))
            print("\t\t\t G: {:7.4f}, reg: {:7.4f} \t\t Fake: {:7.4f}, Real: {:7.4f}".format(loss_g_array[epoch], loss_reg_array[epoch], loss_d_fake_array[epoch], loss_d_real_array[epoch]))
            sys.stdout.flush()

            if plot_metadata_training:
                n_samples = 64
                noise = torch.randn((n_samples, noise_size[0], noise_size[1]), device=device)
                with torch.no_grad():
                    generated_samples = generator(noise)
                    np.savez(out_dir+"/samples_epoch_{:d}.npz".format(epoch), generated_samples.cpu().detach().numpy())
     
    end_time = time()
    print("Total time elapsed for training:", end_time - start_time)

    n_samples = 64 # Generate 64 samples
    noise = torch.randn((n_samples, noise_size[0], noise_size[1]), device=device)
    with torch.no_grad():
        generated_samples = generator(noise)

    np.savez(out_dir+"/samples.npz", generated_samples.cpu().detach().numpy())

    np.savez(out_dir+"/metaEvo.npz", \
            loss_d_fake = loss_d_fake_array.cpu().detach().numpy(), \
            loss_d_real = loss_d_real_array.cpu().detach().numpy(), \
            loss_d = loss_d_array.cpu().detach().numpy(), \
            acc_d_fake = acc_d_fake_array.cpu().detach().numpy(), \
            acc_d_real = acc_d_real_array.cpu().detach().numpy(), \
            loss_g = loss_g_array.cpu().detach().numpy(), \
            loss_reg = loss_reg_array.cpu().detach().numpy(), \
            loss_gen=loss_gen_array.cpu().detach().numpy())

    torch.save(generator.state_dict(), out_dir + '/generator.pt')
    torch.save(discriminator.state_dict(), out_dir + '/discriminator.pt')

    with open( os.path.join(out_dir, "time.txt"), "w") as f:
        f.write("Total time to train in seconds: {:f}".format(end_time - start_time))
    
    return 

# nohup python3 train_model.py > nohup_1.out &
if __name__ == '__main__':
    lr = 0.002
    epochs = 250
    batch_size = 16
    k_epochs_d = 2
    k_epochs_g = 1

    out_dir = './generated'
    alpha = 0.0 # regularization parameter
    beta = 1.0 # generator loss multiplier 0.5
    gamma = 1.0 # discriminator loss multiplier 3.0
    edge = -1 # Deprecreated

    data_type = "full" # full samples from the original data 
    data_stride = 2**15
    len_samples = 2**15
    
    # out_dir = ut.get_dir(out_dir)
    out_dir = os.path.join(out_dir, 'lkfQq8')
    
    weights = torch.Tensor([1,1,0.5,0.5,0.5,0.5,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25])
    print(out_dir)

    meta_dict = {
        "lr":lr,
        "epochs":epochs,
        "batch_size":batch_size,
        "k_epochs_d":k_epochs_d,
        "k_epochs_g":k_epochs_g,
        "out_dir":out_dir,
        "alpha":alpha,
        "beta":beta,
        "gamma":gamma,
        "weights":weights,
        "data_type_loading":data_type,
        "data_type_stride":data_stride,
        "len_samples":len_samples,
        "train_file_type": "Training done for a combined discriminator loss",
    }

    ut.save_meta(meta_dict, out_dir)
    train_model(lr, epochs, batch_size, k_epochs_d, k_epochs_g, alpha, beta, gamma, weights, data_type, data_stride, len_samples, out_dir)


