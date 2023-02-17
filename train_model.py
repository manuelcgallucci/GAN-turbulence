# -*- coding: utf-8 -*-
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
import matplotlib.pyplot as plt

import dataloader as dl
import nn_definitions as nn_d
import utility as ut
from model_3 import CNNGenerator

def train_model( lr, epochs, batch_size, k_epochs_d, alpha, beta, gamma, out_dir, noise_size=(1,2**15)):
    
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

    discriminator = nn_d.Discriminator().to(device)
    discriminator = discriminator.apply(nn_d.weights_init)

    # define loss and optimizers
    criterion_BCE = torch.nn.BCELoss().to(device)
    criterion_MSE = torch.nn.MSELoss().to(device)
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_g = torch.optim.Adam(generator.parameters(),lr= lr, betas=(0.5, 0.999))

    # Train dataset 
    data_train = torch.Tensor(np.genfromtxt('data_train.csv', delimiter=',')) # Nsamples x L (L: length)
    data_samples = data_train.size()[0]
    data_len = data_train.size()[1]

    train_set = dl.DatasetCustom(data_train)
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
    
    s2 = ut.calculate_s2(torch.cumsum(data_train[:,None,:], dim=2), scales, device=device)
    mean_s2 = torch.mean(s2, dim=[0,1]) # This gives the s2 tensor of size (len(scales))
    mean_s2 = mean_s2[None, None, :] 
    for epoch in range(epochs):
        
        running_loss_gen = 0 
        running_loss_g = 0
        running_loss_reg = 0

        running_loss_disc = 0
        running_loss_fake = 0
        running_loss_real = 0

        running_acc_fake_disc = 0
        running_acc_real_disc = 0

        for batch_idx, data_ in enumerate(train_loader):

            data_ = data_.to(device).float()
            # data_ = torch.unsqueeze(data_, dim=1)
            batch_size_ = data_.shape[0]
                
            target_ones = torch.ones((batch_size_, 1), device=device)
            target_zeros = torch.zeros((batch_size_, 1), device=device)

            ## TRAIN DISCRIMINATOR
            for k in range(k_epochs_d):
                discriminator.zero_grad()
                
                # optim_d.zero_grad()
                ## True samples
                pred_real = discriminator(data_)
                loss_real = criterion_BCE(pred_real, target_ones)

                ## False samples (Create random noise and run the generator on them)
                noise = torch.randn((batch_size_, noise_size[0], noise_size[1]), device=device)
                with torch.no_grad():
                    fake_samples = generator(noise)
                                    
                pred_fake = discriminator(fake_samples)
                loss_fake = criterion_BCE(pred_fake, target_zeros)
                
                # Combine the losses 
                loss_d = gamma * (loss_real + loss_fake) / 2
                
                # loss_real.backward()
                # loss_fake.backward()
                loss_d.backward()

                # Discriminator optimizer step
                optim_d.step()
                
                running_loss_real += loss_real / k_epochs_d
                running_loss_fake += loss_fake / k_epochs_d
                running_loss_disc += loss_d / k_epochs_d

            # If pred_fake is all zeros then acc should be 1.0
            # We want this to be around 0.5. 1.0 means perfect accuracy (the generated samples are not similar to the samples)
            running_acc_fake_disc += torch.sum(pred_fake < 0.5)
            running_acc_real_disc += torch.sum(pred_real >= 0.5)

            #optim_g.zero_grad()
            ## TRAIN GENERATOR
            generator.zero_grad()

            noise = torch.randn((batch_size_, noise_size[0], noise_size[1]), device=device)
            generated_signal = generator(noise) 
            # Cut samples (no)
            
            classifications = discriminator(generated_signal)
            
            loss_g = criterion_BCE(classifications, target_ones)

            # E [( X * cumsum(Z) ) ^2]
            # loss_reg = torch.mean(torch.square(torch.mul(noise,torch.cumsum(generated_signal, dim=2))))
            # E[X * cumusm(z)]
            # loss_reg = torch.mean(torch.mul(noise,torch.cumsum(generated_signal, dim=2)))
            # E[Z]^2
            # loss_reg = torch.square(torch.mean(generated_signal) * batch_size_)
            
            # Structure function loss
            # Broadcast s2 into all structure function estimations
            generated_s2 = ut.calculate_s2_v2(torch.cumsum(generated_signal,dim=2), scales, device=device)
            loss_reg = criterion_MSE(generated_s2, mean_s2)

            loss_gen = beta * ( alpha_comp * loss_g + alpha *  loss_reg )
            
            running_loss_gen += loss_gen.item()
            running_loss_reg += loss_reg.item()
            running_loss_g += loss_g.item()

            loss_gen.backward()
            optim_g.step()
            
        # THESE HAVE TO BE DEVIDED BY THE NUMBER OF BATCHES
        loss_g_array[epoch] = running_loss_g
        loss_reg_array[epoch] = running_loss_reg
        loss_gen_array[epoch] = running_loss_gen           

        loss_d_fake_array[epoch] = running_loss_fake
        loss_d_real_array[epoch] = running_loss_real
        loss_d_array[epoch] = running_loss_disc

        acc_d_fake_array[epoch] = running_acc_fake_disc / data_samples
        acc_d_real_array[epoch] = running_acc_real_disc / data_samples

        if epoch%1 == 0:
            print('Epoch [{}/{}] -\t Generator Loss: {:7.4f} \t/\t\t Discriminator Loss: {:7.4f} || acc(fake): {:7.4f} , acc(real): {:7.4f} ||'.format(epoch+1, epochs, running_loss_gen, running_loss_disc, acc_d_fake_array[epoch], acc_d_real_array[epoch]))
            print("\t\t\t G: {:7.4f}, reg: {:7.4f} \t\t Fake: {:7.4f}, Real: {:7.4f}".format(running_loss_g, running_loss_reg, running_loss_fake, running_loss_real))

    n_samples = 64 # Generate 64 samples
    noise = torch.randn((n_samples, noise_size[0], noise_size[1]), device=device)
    with torch.no_grad():
        generated_samples = generator(noise)

    np.savez(out_dir+"samples.npz", generated_samples.cpu().detach().numpy())

    np.savez(out_dir+"metaEvo.npz", \
            loss_d_fake = loss_d_fake_array.cpu().detach().numpy(), \
            loss_d_real = loss_d_real_array.cpu().detach().numpy(), \
            loss_d = loss_d_array.cpu().detach().numpy(), \
            acc_d_fake = acc_d_fake_array.cpu().detach().numpy(), \
            acc_d_real = acc_d_real_array.cpu().detach().numpy(), \
            loss_g = loss_g_array.cpu().detach().numpy(), \
            loss_reg = loss_reg_array.cpu().detach().numpy(), \
            loss_gen=loss_gen_array.cpu().detach().numpy())

    torch.save(generator.state_dict(), out_dir + 'generator.pt')
    torch.save(discriminator.state_dict(), out_dir + 'discriminator.pt')

if __name__ == '__main__':
    lr = 0.002
    epochs = 100
    batch_size = 32
    k_epochs_d = 2
    
    out_dir = './gen_data/'
    alpha = 0.5 # regularization parameter
    beta = 0.01 # generator loss multiplier
    gamma = 1.0 # discriminator loss multiplier

    meta_dir = {
        "lr":lr,
        "epochs":epochs,
        "batch_size":batch_size,
        "k_epochs_d":k_epochs_d,
        "out_dir":out_dir,
        "alpha":alpha,
        "beta":beta,
        "gamma":gamma
    }

    train_model(lr, epochs, batch_size, k_epochs_d, alpha, beta, gamma, out_dir)


