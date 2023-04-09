# from model_discriminator import DiscriminatorMultiNet as Discriminator
import nn_definitions as nn_d
import torch
from torchsummary import summary

def print_exo(x, pot):
    if pot > 4:
        return [""]
    len_x = x.size()[2]
    k = 0
    incrs = int(x.size()[2] / 2**pot)
    #print("\tFor: ", 2**(15-pot))
    print("")
    i = 0
    cat_line = []
    while k <= len_x:
        
        next_k = int(k + x.size()[2] / 2**pot)

        if k == 0:
            print("l{:d}_{:d} = self.cnn{:d}(x[:,:,:{:d}])".format(incrs, i, incrs,next_k))#, "len:", x[:,:,:next_k].size()[2])
        elif k == len_x:
            print("l{:d}_{:d} = self.cnn{:d}(x[:,:,{:d}:])".format(incrs, i, incrs,k))#, "len:", x[:,:,k:].size()[2])
        else:
            print("l{:d}_{:d} = self.cnn{:d}(x[:,:,{:d}:{:d}])".format(incrs, i, incrs,k, next_k))#, "len:", x[:,:,k:next_k].size()[2])
        
        k = next_k
        i = i + 1
	
    all_l = ["l{:d}_{:d}".format(incrs, j) for j in range(0, i-1)]
    print(*all_l)
    print_exo(x, pot+1)
    #print(all_l)
    return  


x = torch.randn((2, 1, 2**16))
all_l = print_exo(x, 2)

# print(all_l)
# device = "cuda"
# discriminator = Discriminator().to(device)
# discriminator = discriminator.apply(nn_d.weights_init)


# noise_size = (1, 2**16)
# batch_size = 8

# noise = torch.randn((batch_size, noise_size[0], noise_size[1]), device=device)


# outs = discriminator(noise)

# print(outs)
# summary(discriminator, noise_size)
