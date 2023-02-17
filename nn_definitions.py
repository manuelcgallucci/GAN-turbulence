from torch import nn
from functools import partial
from numpy import floor
import torch


class Generator(nn.Module):
    # The signals are of size Nbatch x 1 x 2**15
    def __init__(self, l_in):
        super().__init__()
        
        kernels = [2,8,16,32]
        strides = [2,4,8,8]
        avg_pool = [2,2,2,2]
        stride_pool = [1,1,1,1]
        n_ftrs = [1,16,32,64,128]

        self.n_kernels = len(kernels)
        upsample_out_size = []
        self.conv_layers = []
        l_out = l_in
        for i in range(self.n_kernels):
            modules = []

            upsample_out_size.append(l_out)

            modules.append(nn.Conv1d(n_ftrs[i], n_ftrs[i+1], kernel_size = kernels[i], stride = strides[i], padding = 0, bias = False))
            l_out = int(floor(1 + ( (l_out + 2*0 - 1 * (kernels[i] - 1) - 1)/ strides[i]))) # No padding             

            modules.append(nn.BatchNorm1d(n_ftrs[i+1]))
            modules.append(nn.ReLU())
            modules.append(nn.AvgPool1d(kernel_size=avg_pool[i], stride=stride_pool[i]))
            l_out = int(floor(1 + ( (l_out - avg_pool[i])/ stride_pool[i])))

            self.conv_layers.append( nn.Sequential( *modules))

        self.convT_layers = []
        for i in reversed(range(self.n_kernels)):
            modules = []

            # This breakes the forward with "TypeError: forward() missing 1 required positional argument: 'indices'"
            # modules.append(nn.MaxUnpool1d(kernel_size=avg_pool[i], stride=stride_pool[i]))
            
            # Change the upscale value and then compensate with padding and out_padding in the ConvTranspose so that the res connections have the same size 
            modules.append(nn.Upsample(scale_factor=(floor((upsample_out_size[i] + 1 - kernels[i]) / strides[i]) + 1)/l_out))
            print((floor((upsample_out_size[i] + 1 - kernels[i]) / strides[i]) + 2)/l_out)
            l_out = floor(l_out * (floor((upsample_out_size[i] + 1 - kernels[i]) / strides[i]) + 2)/l_out)
            

            # Output size of the convolution with no padding or output_padding 
            l_out = (l_out - 1) * strides[i] - 2*0 + 1* (kernels[i] - 1) + 0 + 1
            
            print(l_out, upsample_out_size[i])
            assert l_out > upsample_out_size[i], "ValueError: Out size cannot be smaller than the expected output"
            if ((l_out - upsample_out_size[i]) % 2) == 0:
                pad_ = int((l_out - upsample_out_size[i]) / 2)
                out_pad_ = 0
            else:
                pad_ = int((l_out - upsample_out_size[i]) / 2) + 1
                out_pad_ = 1
            modules.append(nn.ConvTranspose1d(n_ftrs[i+1], n_ftrs[i], kernel_size = kernels[i], stride = strides[i], padding = pad_, output_padding=out_pad_, bias = False))

            l_out = l_out - 2*pad_ + out_pad_
            
            print(l_out, upsample_out_size[i])
            modules.append(nn.BatchNorm1d(n_ftrs[i]))
            modules.append(nn.ReLU())
            
            self.convT_layers.append( nn.Sequential( *modules))
        print(upsample_out_size)
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.convT_layers = nn.ModuleList(self.convT_layers)
    
    def forward(self, x):  
    
        print(x.shape)
        for i in range(self.n_kernels):
            x = self.conv_layers[i](x)
            print(x.shape)
        
        for i in range(self.n_kernels):
            x = self.convT_layers[i](x)
            print(x.shape)
        
        return x

        """
        res1 = self.conv_layers[0](x)
        res2 = self.conv_layers[1](res1)
        res3 = self.conv_layers[2](res2)
        out = self.conv_layers[3](res3)
        
        x = self.convT_layers[0](out)
        x = self.convT_layers[1](torch.add(x, res3))
        x = self.convT_layers[2](torch.add(x, res2))
        x = self.convT_layers[3](torch.add(x, res1))
        """
        
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size = 8, stride = 4, padding = 0, bias = False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(4, 8, kernel_size = 8, stride = 4, padding = 0, bias = False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(8, 16, kernel_size = 5, stride = 3, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(16, 16, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(16, 32, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(32, 64, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(64, 64, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(64, 64, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Flatten(),
            
            nn.Linear(64*20, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        
        #out = out.squeeze(1)
        return self.cnn(x)

def weights_init(m):
    """
    This function initializes the model weights randomly from a 
    Normal distribution. This follows the specification from the DCGAN paper.
    https://arxiv.org/pdf/1511.06434.pdf
    Source: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class CNNGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # The input of my generator is a noise of size (2048,1)
        # The output of my generator is a signal of size (2048,1)
        self.avgpool = nn.AvgPool1d(2, ceil_mode=False)
        self.avgpoolc = nn.AvgPool1d(2, ceil_mode=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear')
        #self.slopeleak= 0.005
        self.cnn1 = nn.Sequential( 
            nn.Conv1d(1, 16, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            #nn.LeakyReLU(self.slopeleak,True),
            nn.ReLU(True),
            )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size = 2, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            #nn.LeakyReLU(self.slopeleak,True),
            nn.ReLU(True),
            )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(64),
            #nn.LeakyReLU(self.slopeleak,True),
            nn.ReLU(True),
            )
        self.cnn8 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size = 8, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(128),
            #nn.LeakyReLU(self.slopeleak,True),
            nn.ReLU(True),
            )
        self.cnn16 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size = 16, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            #nn.LeakyReLU(self.slopeleak,True),
            nn.ReLU(True),
            )
        self.cnn32 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size = 32, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            #nn.LeakyReLU(self.slopeleak,True),
            nn.ReLU(True),
            )
        self.cnn64 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size = 64, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            #nn.LeakyReLU(self.slopeleak,True),
            nn.ReLU(True),
            )
        self.cnntrans64 = nn.Sequential(
            nn.ConvTranspose1d(256, 256, kernel_size = 64, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            #nn.LeakyReLU(self.slopeleak,True),
            nn.ReLU(True),
            )
        self.cnntrans32 = nn.Sequential(
            nn.ConvTranspose1d(256, 256, kernel_size = 32, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            #nn.LeakyReLU(self.slopeleak,True),
            nn.ReLU(True),
            )
        self.cnntrans16 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size = 16, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(128),
            #nn.LeakyReLU(self.slopeleak,True),
            nn.ReLU(True),
            )
        self.cnntrans8 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size = 8, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(64),
            #nn.LeakyReLU(self.slopeleak,True),
            nn.ReLU(True),
            )
        self.cnntrans4 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            #nn.LeakyReLU(self.slopeleak,True),
            nn.ReLU(True),
            )
        self.cnntrans2 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size = 2, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            #nn.LeakyReLU(self.slopeleak,True),
            nn.ReLU(True),
            )
        self.cnntrans1 = nn.Sequential(
            nn.ConvTranspose1d(16, 1, kernel_size = 1, stride = 1, padding = 0, bias = False),
            #nn.Tanh()
            )
        self.bridge1 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size = 64, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            #nn.LeakyReLU(self.slopeleak,True),
            nn.ReLU(True),
            )
        self.bridge2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size = 128, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            #nn.LeakyReLU(self.slopeleak,True),
            nn.ReLU(True),
            )
        self.bridge3 = nn.Sequential(
            nn.ConvTranspose1d(256, 256, kernel_size = 128, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            #nn.LeakyReLU(self.slopeleak,True),
            nn.ReLU(True),
            )
        self.bridge4 = nn.Sequential(
            nn.ConvTranspose1d(256, 256, kernel_size = 64, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            #nn.LeakyReLU(self.slopeleak,True),
            nn.ReLU(True),
            )
        
    def forward(self, z):    
        residual1  = self.cnn1(z)
        out = residual1 #= out->  size
        #print(out.shape)
        
        #out = self.avgpoolc(out)
        residual2  = self.cnn2(out)
        #out += residual1[:,:,0:len(out)]
        out = residual2 #= out -> size/2
        #print(out.shape)
        
        out = self.avgpoolc(out)
        residual4  = self.cnn4(out)
        #out += residual2[:,:,0:len(out)]
        out = residual4 #= out ->size/4
        #print(out.shape)
        
        out = self.avgpoolc(out)
        residual8  = self.cnn8(out)
        #out += residual4[:,:,0:len(out)]
        out = residual8 #= out -> size=8
        #print(out.shape)
        
        out = self.avgpoolc(out)
        residual16  = self.cnn16(out)
        #out += residual8[:,:,0:len(out)]
        out = residual16 #= out -> size/16
        #print(out.shape)
        
        out = self.avgpoolc(out)
        residual32  = self.cnn32(out)
        #out += residual16[:,:,0:len(out)]
        out = residual32 #= out size/32
        #print(out.shape)
        
        out = self.avgpoolc(out)
        out  = self.cnn64(out)
        #out += residual32[:,:,0:len(out)]
        #residual64 = out -> size/64
        #print(out.shape)
        
        #Bridge
        out = self.bridge1(out)
        #out = self.bridge2(out)
        #out = self.bridge3(out)
        out = self.bridge4(out)
        #End of Bridge
        
        out  = self.cnntrans64(out)
        out = self.upsample(out)
        #print(out.shape)
        out = out + residual32 # -> size/32
        
        out  = self.cnntrans32(out)
        out  =self.upsample(out)
        #print(out.shape)
        out = out[:,:,0:-1] + residual16 # -> size/16
        
        out  = self.cnntrans16(out)
        out  =self.upsample(out)
        #print(out.shape)
        out  = out + residual8 #-> size/8

        out  = self.cnntrans8(out)
        out  =self.upsample(out)
        #print(out.shape)
        out  = out[:,:,0:-1] + residual4 #-> size/4

        out  = self.cnntrans4(out)
        out  =self.upsample(out)
        #print(out.shape)
        out  = out[:,:,0:-1] + residual2 #-> size/2

        out  = self.cnntrans2(out)
        #out  =self.upsample(out)
        #print(out.shape)
        out  = out + residual1 #->size

        out  = self.cnntrans1(out)
        return out

class DiscriminatorLinear(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size = 8, stride = 4, padding = 0, bias = False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(4, 8, kernel_size = 8, stride = 4, padding = 0, bias = False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(8, 16, kernel_size = 5, stride = 3, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(16, 16, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(16, 32, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(32, 64, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(64, 64, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(64, 64, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Flatten(),
            
            nn.Linear(64*20, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 1),
        )
        
    def forward(self, x):
        
        #out = out.squeeze(1)
        return self.cnn(x)
