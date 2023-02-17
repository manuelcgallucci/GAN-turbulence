import torch
import numpy as np 
import matplotlib.pyplot as plt

import utility as ut
from model_3 import CNNGenerator

data_dir = './gen_data/'
save = True
display = False

def main():
    # plot_real_data()
    
    plot_structure()
    plot_history()
    plot_samples()
    
def plot_history():

    history = np.load(data_dir + 'metaEvo.npz')

    plt.figure()
    plt.plot(history["loss_d_real"])
    plt.plot(history["loss_d_fake"])
    plt.plot(history["loss_d"])
    plt.title("Losses of Discriminator")
    plt.legend(["Real", "Fake", "Total"])
    if save: plt.savefig(data_dir + "loss_discriminator.png")
    if display: plt.show()

    plt.figure()
    plt.plot(history["loss_g"] / np.max(history["loss_g"]))
    plt.plot(history["loss_reg"] / np.max(history["loss_reg"]))
    plt.plot(history["loss_gen"] / np.max(history["loss_gen"]))
    plt.title("Losses of Generator -Normalized-")
    plt.legend(["Generator", "Regularization", "Total"])
    if save: plt.savefig(data_dir + "loss_generator.png")
    if display: plt.show()

    plt.figure()
    plt.plot(history["loss_gen"])
    plt.plot(history["loss_d"])
    plt.title("Losses of the models")
    plt.legend(["Generator", "Discriminator"])
    if save: plt.savefig(data_dir + "losses.png")
    if display: plt.show()

    plt.figure()
    plt.plot(history["acc_d_fake"])
    plt.plot(history["acc_d_real"])
    plt.title("Discriminator Accuracy on generated samples")
    plt.legend(["Fake samples", "Real samples"])
    if save: plt.savefig(data_dir + "accuracy.png")
    if display: plt.show()

    plt.figure()
    plt.plot(history["loss_reg"])
    plt.title("Loss Regularization")
    if save: plt.savefig(data_dir + "loss_regularization.png")
    if display: plt.show()

### ======================================= ###
def plot_samples():
    print("\tLoading samples ... ")
    samples = np.load(data_dir + 'samples.npz')["arr_0"]

    fig = plt.figure()
    for k in range(samples.shape[0]):
        plt.plot(np.cumsum(samples[k, 0, :]), linewidth=1.0)
    fig.suptitle('Generated cumsum samples')
    if save: plt.savefig(data_dir + "sample_once.png")
    if display: plt.show()

    fig, ax = plt.subplots(8,8)
    for k, axs in enumerate(fig.axes):
        axs.plot(np.cumsum(samples[k, 0, :]))
        axs.get_xaxis().set_ticks([])
        axs.get_yaxis().set_ticks([])
    if save: plt.savefig(data_dir + "sample_all.png")
    if display: plt.show()


### ======================================= ###
def plot_structure(n_samples=64, len_=2**15, edge=4096, device="cuda"):
    color=np.array([166, 178, 255])/255.0

    nv=10
    uu=2**np.arange(0,13,1/nv)
    scales=np.unique(uu.astype(int))
    scales=scales[0:100]

    generator = CNNGenerator().to(device)
    generator.load_state_dict(torch.load(data_dir + 'generator.pt'))
    noise = torch.randn((n_samples, 1, len_+2*edge), device=device)
    with torch.no_grad():
        generated_samples = generator(noise)
    
    plt.figure()
    for k in range(n_samples):
        plt.plot(generated_samples[k,0,:].cpu(), linewidth=1.0)
    plt.title("Generated samples")
    if save: plt.savefig(data_dir + "s2_gen_samples.png")
    if display: plt.show()
    
    s2 = ut.calculate_s2(generated_samples, scales, device=device).cpu()

    plt.figure()
    for k in range(n_samples):
        plt.plot(np.log(scales), s2[k,0,:], color=color, linewidth=1.0)
    plt.plot(np.log(scales), torch.mean(s2[:,0,:], dim=0), 'r', linewidth=2.0)
    plt.title("Structure function on the samples")
    plt.xlabel("scales (log)")
    if save: plt.savefig(data_dir + "s2_samples.png")
    if display: plt.show()

    generated_samples = torch.cumsum(generated_samples, dim=2)[:,:,edge:-edge]

    plt.figure()
    for k in range(n_samples):
        plt.plot(generated_samples[k,0,:].cpu(), linewidth=1.0)
    plt.title("Generated cumsum samples")
    if save: plt.savefig(data_dir + "s2_gen_samples_cumsum.png")
    if display: plt.show()

    s2 = ut.calculate_s2(generated_samples, scales, device=device).cpu()

    plt.figure()
    for k in range(n_samples):
        plt.plot(np.log(scales), s2[k,0,:], color=color, linewidth=1.0)
    plt.plot(np.log(scales), torch.mean(s2[:,0,:], dim=0), 'r', linewidth=2.0)
    plt.title("Structure function on the cumsum samples")
    plt.xlabel("scales (log)")
    if save: plt.savefig(data_dir + "s2_cumsum_samples.png")
    if display: plt.show()

### ======================================= ###

def plot_real_data():
    data_train = np.genfromtxt('data_train.csv', delimiter=',')

    fig, ax = plt.subplots(8,8)
    fig.suptitle('Real cumsum samples (Train)')
    for k, axs in enumerate(fig.axes):
        axs.plot(np.cumsum(data_train[k, :]))
        axs.get_xaxis().set_ticks([])
        axs.get_yaxis().set_ticks([])
    if save: plt.savefig(data_dir + "samples_real_cumusm.png")
    if display: plt.show()

    plt.figure()
    max_k = 64
    for k in range(data_train.shape[0]):
        if k >= max_k: break
        plt.plot(np.cumsum(data_train[k, :]), linewidth=1.0)
    fig.suptitle(str(max_k) + 'Real cumsum samples (Train)')
    if save: plt.savefig(data_dir + "sample_once_real_cumsum.png")
    if display: plt.show()



if __name__ == '__main__':
    main()