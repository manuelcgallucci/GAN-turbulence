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

import dataloader as dl
import nn_definitions as nn_d
import utility as ut
from model_generator import CNNGeneratorBigConcatDropout as CNNGenerator


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
    data_train = torch.Tensor(np.load('./data/data.npy')) # Nsamples x L (L: length)
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
    
    # s2 = ut.calculate_s2(torch.cumsum(data_train[:,None,:], dim=2), scales, device=device)
    s2 = ut.calculate_s2(data_train[:,None,:], scales, device=device)
    mean_s2 = torch.mean(s2, dim=[0,1]) # This gives the s2 tensor of size (len(scales))
    mean_s2 = mean_s2[None, None, :] 

    # Take the target ones and zeros for the batch size and for the last (not complete batch)

    target_ones_full = torch.ones((batch_size, 1), device=device)
    target_ones_partial = torch.ones((data_samples - batch_size * int(data_samples / batch_size), 1), device=device)
    target_ones = [target_ones_full, target_ones_partial]

    target_zeros_full = torch.zeros((batch_size, 1), device=device)
    target_zeros_partial = torch.ones((data_samples - batch_size * int(data_samples / batch_size), 1), device=device)
    target_zeros = [target_zeros_full, target_zeros_partial]
    
    last_batch_idx = np.ceil(data_samples / batch_size) - 1
    
    start_time = time()
    for epoch in range(epochs):

        for batch_idx, data_ in enumerate(train_loader):

            data_ = data_.to(device).float()
            # data_ = torch.unsqueeze(data_, dim=1)
            batch_size_ = data_.shape[0]

            ## TRAIN DISCRIMINATOR
            for k in range(k_epochs_d):
                discriminator.zero_grad()
                
                # optim_d.zero_grad()
                ## True samples
                pred_real = discriminator(data_)
                loss_real = criterion_BCE(pred_real, target_ones[batch_idx == last_batch_idx])

                ## False samples (Create random noise and run the generator on them)
                noise = torch.randn((batch_size_, noise_size[0], noise_size[1]), device=device)
                with torch.no_grad():
                    fake_samples = generator(noise)
                                    
                pred_fake = discriminator(fake_samples)
                loss_fake = criterion_BCE(pred_fake, target_zeros[batch_idx == last_batch_idx])
                
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

            # If pred_fake is all zeros then acc should be 1.0
            # We want this to be around 0.5. 1.0 means perfect accuracy (the generated samples are not similar to the samples)
            acc_d_fake_array[epoch] += torch.sum(pred_fake < 0.5).item()
            acc_d_real_array[epoch] += torch.sum(pred_real >= 0.5).item()

            #optim_g.zero_grad()
            ## TRAIN GENERATOR
            generator.zero_grad()

            noise = torch.randn((batch_size_, noise_size[0], noise_size[1]), device=device)
            generated_signal = generator(noise) 
            # Cut samples (no)
            
            classifications = discriminator(generated_signal)
            loss_g = criterion_BCE(classifications, target_ones[batch_idx == last_batch_idx])

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

            loss_gen_array[epoch] += loss_gen.item()
            # loss_reg_array[epoch] += loss_reg.item()
            loss_g_array[epoch] += loss_g.item()

            loss_gen.backward()
            optim_g.step()

        # THESE HAVE TO BE DEVIDED BY THE NUMBER OF BATCHES
        acc_d_fake_array[epoch] = acc_d_fake_array[epoch] / data_samples
        acc_d_real_array[epoch] = acc_d_real_array[epoch] / data_samples

        if epoch%1 == 0:
            print('Epoch [{}/{}] -\t Generator Loss: {:7.4f} \t/\t\t Discriminator Loss: {:7.4f} || acc(fake): {:7.4f} , acc(real): {:7.4f} ||'.format(epoch+1, epochs, loss_gen_array[epoch], loss_d_array[epoch], acc_d_fake_array[epoch], acc_d_real_array[epoch]))
            print("\t\t\t G: {:7.4f}, reg: {:7.4f} \t\t Fake: {:7.4f}, Real: {:7.4f}".format(loss_g_array[epoch], loss_reg_array[epoch], loss_d_fake_array[epoch], loss_d_real_array[epoch]))

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
        f.write("\nTotal time to train in seconds: {:f}".format(end_time - start_time))
    
    return 

if __name__ == '__main__':
    lr = 0.002
    epochs = 400
    batch_size = 32
    k_epochs_d = 3
    
    out_dir = './generated'
    alpha = 0.0 # regularization parameter
    beta = 0.5 # generator loss multiplier
    gamma = 1.0 # discriminator loss multiplier
    
    out_dir = ut.get_dir(out_dir)
    print(out_dir)
    # out_dir = os.path.join(out_dir, 'ILUyQ7')

    meta_dict = {
        "lr":lr,
        "epochs":epochs,
        "batch_size":batch_size,
        "k_epochs_d":k_epochs_d,
        "out_dir":out_dir,
        "alpha":alpha,
        "beta":beta,
        "gamma":gamma
    }

    ut.save_meta(meta_dict, out_dir)
    train_model(lr, epochs, batch_size, k_epochs_d, alpha, beta, gamma, out_dir)


