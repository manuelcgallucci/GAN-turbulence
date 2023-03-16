from torch import nn
import torch




###  Discriminator Multiresolution
### From the 1st model
class DiscriminatorMultiNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn8192 = nn.Sequential(
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

			nn.Linear(64*4, 64),
			nn.LeakyReLU(0.2),

			nn.Linear(64, 1)
		)

        self.cnn4096 = nn.Sequential(
			nn.Conv1d(1, 4, kernel_size = 8, stride = 4, padding = 0, bias = False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(4, 8, kernel_size = 5, stride = 3, padding = 0, bias = False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(8, 16, kernel_size = 5, stride = 2, padding = 0, bias = False),
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
            
            nn.Linear(64*9, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 1)
		)

        self.cnn2048 = nn.Sequential(
			nn.Conv1d(1, 4, kernel_size = 5, stride = 3, padding = 0, bias = False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(4, 8, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(8, 16, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(16, 16, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(16, 32, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(32, 64, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Flatten(),
            
            nn.Linear(64*20, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 1)
		)

        self.cnn1024 = nn.Sequential(
			nn.Conv1d(1, 4, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(4, 8, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(8, 16, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(16, 32, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(32, 32, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Flatten(),
            
            nn.Linear(32*30, 32),
            nn.LeakyReLU(0.2),

            nn.Linear(32, 1)
		)

        self.cnn512 = nn.Sequential(
			nn.Conv1d(1, 4, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(4, 8, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(8, 16, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(16, 32, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(32, 32, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Flatten(),
            
            nn.Linear(32*14, 16),
            nn.LeakyReLU(0.2),

            nn.Linear(16, 1)
		)

        self.endDense = nn.Sequential(
			nn.Linear(124, 1),
			nn.Sigmoid()
		)
		
    def forward(self, x):

        l8192_0 = self.cnn8192(x[:,:,:8192])
        l8192_1 = self.cnn8192(x[:,:,8192:16384])
        l8192_2 = self.cnn8192(x[:,:,16384:24576])
        l8192_3 = self.cnn8192(x[:,:,24576:])


        l4096_0 = self.cnn4096(x[:,:,:4096])
        l4096_1 = self.cnn4096(x[:,:,4096:8192])
        l4096_2 = self.cnn4096(x[:,:,8192:12288])
        l4096_3 = self.cnn4096(x[:,:,12288:16384])
        l4096_4 = self.cnn4096(x[:,:,16384:20480])
        l4096_5 = self.cnn4096(x[:,:,20480:24576])
        l4096_6 = self.cnn4096(x[:,:,24576:28672])
        l4096_7 = self.cnn4096(x[:,:,28672:])


        l2048_0 = self.cnn2048(x[:,:,:2048])
        l2048_1 = self.cnn2048(x[:,:,2048:4096])
        l2048_2 = self.cnn2048(x[:,:,4096:6144])
        l2048_3 = self.cnn2048(x[:,:,6144:8192])
        l2048_4 = self.cnn2048(x[:,:,8192:10240])
        l2048_5 = self.cnn2048(x[:,:,10240:12288])
        l2048_6 = self.cnn2048(x[:,:,12288:14336])
        l2048_7 = self.cnn2048(x[:,:,14336:16384])
        l2048_8 = self.cnn2048(x[:,:,16384:18432])
        l2048_9 = self.cnn2048(x[:,:,18432:20480])
        l2048_10 = self.cnn2048(x[:,:,20480:22528])
        l2048_11 = self.cnn2048(x[:,:,22528:24576])
        l2048_12 = self.cnn2048(x[:,:,24576:26624])
        l2048_13 = self.cnn2048(x[:,:,26624:28672])
        l2048_14 = self.cnn2048(x[:,:,28672:30720])
        l2048_15 = self.cnn2048(x[:,:,30720:])


        l1024_0 = self.cnn1024(x[:,:,:1024])
        l1024_1 = self.cnn1024(x[:,:,1024:2048])
        l1024_2 = self.cnn1024(x[:,:,2048:3072])
        l1024_3 = self.cnn1024(x[:,:,3072:4096])
        l1024_4 = self.cnn1024(x[:,:,4096:5120])
        l1024_5 = self.cnn1024(x[:,:,5120:6144])
        l1024_6 = self.cnn1024(x[:,:,6144:7168])
        l1024_7 = self.cnn1024(x[:,:,7168:8192])
        l1024_8 = self.cnn1024(x[:,:,8192:9216])
        l1024_9 = self.cnn1024(x[:,:,9216:10240])
        l1024_10 = self.cnn1024(x[:,:,10240:11264])
        l1024_11 = self.cnn1024(x[:,:,11264:12288])
        l1024_12 = self.cnn1024(x[:,:,12288:13312])
        l1024_13 = self.cnn1024(x[:,:,13312:14336])
        l1024_14 = self.cnn1024(x[:,:,14336:15360])
        l1024_15 = self.cnn1024(x[:,:,15360:16384])
        l1024_16 = self.cnn1024(x[:,:,16384:17408])
        l1024_17 = self.cnn1024(x[:,:,17408:18432])
        l1024_18 = self.cnn1024(x[:,:,18432:19456])
        l1024_19 = self.cnn1024(x[:,:,19456:20480])
        l1024_20 = self.cnn1024(x[:,:,20480:21504])
        l1024_21 = self.cnn1024(x[:,:,21504:22528])
        l1024_22 = self.cnn1024(x[:,:,22528:23552])
        l1024_23 = self.cnn1024(x[:,:,23552:24576])
        l1024_24 = self.cnn1024(x[:,:,24576:25600])
        l1024_25 = self.cnn1024(x[:,:,25600:26624])
        l1024_26 = self.cnn1024(x[:,:,26624:27648])
        l1024_27 = self.cnn1024(x[:,:,27648:28672])
        l1024_28 = self.cnn1024(x[:,:,28672:29696])
        l1024_29 = self.cnn1024(x[:,:,29696:30720])
        l1024_30 = self.cnn1024(x[:,:,30720:31744])
        l1024_31 = self.cnn1024(x[:,:,31744:])


        l512_0 = self.cnn512(x[:,:,:512])
        l512_1 = self.cnn512(x[:,:,512:1024])
        l512_2 = self.cnn512(x[:,:,1024:1536])
        l512_3 = self.cnn512(x[:,:,1536:2048])
        l512_4 = self.cnn512(x[:,:,2048:2560])
        l512_5 = self.cnn512(x[:,:,2560:3072])
        l512_6 = self.cnn512(x[:,:,3072:3584])
        l512_7 = self.cnn512(x[:,:,3584:4096])
        l512_8 = self.cnn512(x[:,:,4096:4608])
        l512_9 = self.cnn512(x[:,:,4608:5120])
        l512_10 = self.cnn512(x[:,:,5120:5632])
        l512_11 = self.cnn512(x[:,:,5632:6144])
        l512_12 = self.cnn512(x[:,:,6144:6656])
        l512_13 = self.cnn512(x[:,:,6656:7168])
        l512_14 = self.cnn512(x[:,:,7168:7680])
        l512_15 = self.cnn512(x[:,:,7680:8192])
        l512_16 = self.cnn512(x[:,:,8192:8704])
        l512_17 = self.cnn512(x[:,:,8704:9216])
        l512_18 = self.cnn512(x[:,:,9216:9728])
        l512_19 = self.cnn512(x[:,:,9728:10240])
        l512_20 = self.cnn512(x[:,:,10240:10752])
        l512_21 = self.cnn512(x[:,:,10752:11264])
        l512_22 = self.cnn512(x[:,:,11264:11776])
        l512_23 = self.cnn512(x[:,:,11776:12288])
        l512_24 = self.cnn512(x[:,:,12288:12800])
        l512_25 = self.cnn512(x[:,:,12800:13312])
        l512_26 = self.cnn512(x[:,:,13312:13824])
        l512_27 = self.cnn512(x[:,:,13824:14336])
        l512_28 = self.cnn512(x[:,:,14336:14848])
        l512_29 = self.cnn512(x[:,:,14848:15360])
        l512_30 = self.cnn512(x[:,:,15360:15872])
        l512_31 = self.cnn512(x[:,:,15872:16384])
        l512_32 = self.cnn512(x[:,:,16384:16896])
        l512_33 = self.cnn512(x[:,:,16896:17408])
        l512_34 = self.cnn512(x[:,:,17408:17920])
        l512_35 = self.cnn512(x[:,:,17920:18432])
        l512_36 = self.cnn512(x[:,:,18432:18944])
        l512_37 = self.cnn512(x[:,:,18944:19456])
        l512_38 = self.cnn512(x[:,:,19456:19968])
        l512_39 = self.cnn512(x[:,:,19968:20480])
        l512_40 = self.cnn512(x[:,:,20480:20992])
        l512_41 = self.cnn512(x[:,:,20992:21504])
        l512_42 = self.cnn512(x[:,:,21504:22016])
        l512_43 = self.cnn512(x[:,:,22016:22528])
        l512_44 = self.cnn512(x[:,:,22528:23040])
        l512_45 = self.cnn512(x[:,:,23040:23552])
        l512_46 = self.cnn512(x[:,:,23552:24064])
        l512_47 = self.cnn512(x[:,:,24064:24576])
        l512_48 = self.cnn512(x[:,:,24576:25088])
        l512_49 = self.cnn512(x[:,:,25088:25600])
        l512_50 = self.cnn512(x[:,:,25600:26112])
        l512_51 = self.cnn512(x[:,:,26112:26624])
        l512_52 = self.cnn512(x[:,:,26624:27136])
        l512_53 = self.cnn512(x[:,:,27136:27648])
        l512_54 = self.cnn512(x[:,:,27648:28160])
        l512_55 = self.cnn512(x[:,:,28160:28672])
        l512_56 = self.cnn512(x[:,:,28672:29184])
        l512_57 = self.cnn512(x[:,:,29184:29696])
        l512_58 = self.cnn512(x[:,:,29696:30208])
        l512_59 = self.cnn512(x[:,:,30208:30720])
        l512_60 = self.cnn512(x[:,:,30720:31232])
        l512_61 = self.cnn512(x[:,:,31232:31744])
        l512_62 = self.cnn512(x[:,:,31744:32256])
        l512_63 = self.cnn512(x[:,:,32256:])


        out = torch.cat((l512_0, l512_1, l512_2, l512_3, l512_4, l512_5, l512_6, l512_7, l512_8, l512_9, l512_10, l512_11, l512_12, l512_13, l512_14, l512_15, l512_16, l512_17, l512_18, l512_19, l512_20, l512_21, l512_22, l512_23, l512_24, l512_25, l512_26, l512_27, l512_28, l512_29, l512_30, l512_31, l512_32, l512_33, l512_34, l512_35, l512_36, l512_37, l512_38, l512_39, l512_40, l512_41, l512_42, l512_43, l512_44, l512_45, l512_46, l512_47, l512_48, l512_49, l512_50, l512_51, l512_52, l512_53, l512_54, l512_55, l512_56, l512_57, l512_58, l512_59, l512_60, l512_61, l512_62, l512_63,  \
            l1024_0, l1024_1, l1024_2, l1024_3, l1024_4, l1024_5, l1024_6, l1024_7, l1024_8, l1024_9, l1024_10, l1024_11, l1024_12, l1024_13, l1024_14, l1024_15, l1024_16, l1024_17, l1024_18, l1024_19, l1024_20, l1024_21, l1024_22, l1024_23, l1024_24, l1024_25, l1024_26, l1024_27, l1024_28, l1024_29, l1024_30, l1024_31,\
            l2048_0, l2048_1, l2048_2, l2048_3, l2048_4, l2048_5, l2048_6, l2048_7, l2048_8, l2048_9, l2048_10, l2048_11, l2048_12, l2048_13, l2048_14, l2048_15,  \
            l4096_0, l4096_1, l4096_2, l4096_3, l4096_4, l4096_5, l4096_6, l4096_7,  \
            l8192_0, l8192_1, l8192_2, l8192_3), dim=1)

        return self.endDense(out)

class DiscriminatorMultiNet16_4(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn16384 = nn.Sequential(
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

			nn.Linear(64*9, 64),
			nn.LeakyReLU(0.2),

			nn.Linear(64, 1),
            nn.Sigmoid()
		)

        self.cnn8192 = nn.Sequential(
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

			nn.Linear(64*4, 64),
			nn.LeakyReLU(0.2),

			nn.Linear(64, 1),
            nn.Sigmoid()
		)

        self.cnn4096 = nn.Sequential(
			nn.Conv1d(1, 4, kernel_size = 8, stride = 4, padding = 0, bias = False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(4, 8, kernel_size = 5, stride = 3, padding = 0, bias = False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(8, 16, kernel_size = 5, stride = 2, padding = 0, bias = False),
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
            
            nn.Linear(64*9, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 1),
            nn.Sigmoid()
		)


		
    def forward(self, x):

        p16384_0 = self.cnn16384(x[:,:,:16384])
        p16384_1 = self.cnn16384(x[:,:,16384:])

        p8192_0 = self.cnn8192(x[:,:,:8192])
        p8192_1 = self.cnn8192(x[:,:,8192:16384])
        p8192_2 = self.cnn8192(x[:,:,16384:24576])
        p8192_3 = self.cnn8192(x[:,:,24576:])

        p4096_0 = self.cnn4096(x[:,:,:4096])
        p4096_1 = self.cnn4096(x[:,:,4096:8192])
        p4096_2 = self.cnn4096(x[:,:,8192:12288])
        p4096_3 = self.cnn4096(x[:,:,12288:16384])
        p4096_4 = self.cnn4096(x[:,:,16384:20480])
        p4096_5 = self.cnn4096(x[:,:,20480:24576])
        p4096_6 = self.cnn4096(x[:,:,24576:28672])
        p4096_7 = self.cnn4096(x[:,:,28672:])

        out = torch.cat((p16384_0, p16384_1, p8192_0, p8192_1, p8192_2, p8192_3, p4096_0, p4096_1, p4096_2, p4096_3, p4096_4, p4096_5, p4096_6, p4096_7),dim=1)
        return out


class DiscriminatorMultiNet16(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn16384 = nn.Sequential(
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

			nn.Linear(64*9, 64),
			nn.Linear(64, 1),
            nn.Sigmoid()
		)

		
    def forward(self, x):

        p16384_0 = self.cnn16384(x[:,:,:16384])
        p16384_1 = self.cnn16384(x[:,:,16384:])

        return p16384_0, p16384_1

class DiscriminatorMultiNetNo512(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn8192 = nn.Sequential(
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

			nn.Linear(64*4, 64),
			nn.LeakyReLU(0.2),

			nn.Linear(64, 1)
		)

        self.cnn4096 = nn.Sequential(
			nn.Conv1d(1, 4, kernel_size = 8, stride = 4, padding = 0, bias = False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(4, 8, kernel_size = 5, stride = 3, padding = 0, bias = False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(8, 16, kernel_size = 5, stride = 2, padding = 0, bias = False),
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
            
            nn.Linear(64*9, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 1)
		)

        self.cnn2048 = nn.Sequential(
			nn.Conv1d(1, 4, kernel_size = 5, stride = 3, padding = 0, bias = False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(4, 8, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(8, 16, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(16, 16, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(16, 32, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(32, 64, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Flatten(),
            
            nn.Linear(64*20, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 1)
		)

        self.cnn1024 = nn.Sequential(
			nn.Conv1d(1, 4, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(4, 8, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(8, 16, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(16, 32, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(32, 32, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Flatten(),
            
            nn.Linear(32*30, 32),
            nn.LeakyReLU(0.2),

            nn.Linear(32, 1)
		)

        self.cnn512 = nn.Sequential(
			nn.Conv1d(1, 4, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(4, 8, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(8, 16, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(16, 32, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(32, 32, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Flatten(),
            
            nn.Linear(32*14, 16),
            nn.LeakyReLU(0.2),

            nn.Linear(16, 1)
		)

        self.endDense = nn.Sequential(
			nn.Linear(60, 1),
			nn.Sigmoid()
		)
		
    def forward(self, x):

        l8192_0 = self.cnn8192(x[:,:,:8192])
        l8192_1 = self.cnn8192(x[:,:,8192:16384])
        l8192_2 = self.cnn8192(x[:,:,16384:24576])
        l8192_3 = self.cnn8192(x[:,:,24576:])


        l4096_0 = self.cnn4096(x[:,:,:4096])
        l4096_1 = self.cnn4096(x[:,:,4096:8192])
        l4096_2 = self.cnn4096(x[:,:,8192:12288])
        l4096_3 = self.cnn4096(x[:,:,12288:16384])
        l4096_4 = self.cnn4096(x[:,:,16384:20480])
        l4096_5 = self.cnn4096(x[:,:,20480:24576])
        l4096_6 = self.cnn4096(x[:,:,24576:28672])
        l4096_7 = self.cnn4096(x[:,:,28672:])


        l2048_0 = self.cnn2048(x[:,:,:2048])
        l2048_1 = self.cnn2048(x[:,:,2048:4096])
        l2048_2 = self.cnn2048(x[:,:,4096:6144])
        l2048_3 = self.cnn2048(x[:,:,6144:8192])
        l2048_4 = self.cnn2048(x[:,:,8192:10240])
        l2048_5 = self.cnn2048(x[:,:,10240:12288])
        l2048_6 = self.cnn2048(x[:,:,12288:14336])
        l2048_7 = self.cnn2048(x[:,:,14336:16384])
        l2048_8 = self.cnn2048(x[:,:,16384:18432])
        l2048_9 = self.cnn2048(x[:,:,18432:20480])
        l2048_10 = self.cnn2048(x[:,:,20480:22528])
        l2048_11 = self.cnn2048(x[:,:,22528:24576])
        l2048_12 = self.cnn2048(x[:,:,24576:26624])
        l2048_13 = self.cnn2048(x[:,:,26624:28672])
        l2048_14 = self.cnn2048(x[:,:,28672:30720])
        l2048_15 = self.cnn2048(x[:,:,30720:])


        l1024_0 = self.cnn1024(x[:,:,:1024])
        l1024_1 = self.cnn1024(x[:,:,1024:2048])
        l1024_2 = self.cnn1024(x[:,:,2048:3072])
        l1024_3 = self.cnn1024(x[:,:,3072:4096])
        l1024_4 = self.cnn1024(x[:,:,4096:5120])
        l1024_5 = self.cnn1024(x[:,:,5120:6144])
        l1024_6 = self.cnn1024(x[:,:,6144:7168])
        l1024_7 = self.cnn1024(x[:,:,7168:8192])
        l1024_8 = self.cnn1024(x[:,:,8192:9216])
        l1024_9 = self.cnn1024(x[:,:,9216:10240])
        l1024_10 = self.cnn1024(x[:,:,10240:11264])
        l1024_11 = self.cnn1024(x[:,:,11264:12288])
        l1024_12 = self.cnn1024(x[:,:,12288:13312])
        l1024_13 = self.cnn1024(x[:,:,13312:14336])
        l1024_14 = self.cnn1024(x[:,:,14336:15360])
        l1024_15 = self.cnn1024(x[:,:,15360:16384])
        l1024_16 = self.cnn1024(x[:,:,16384:17408])
        l1024_17 = self.cnn1024(x[:,:,17408:18432])
        l1024_18 = self.cnn1024(x[:,:,18432:19456])
        l1024_19 = self.cnn1024(x[:,:,19456:20480])
        l1024_20 = self.cnn1024(x[:,:,20480:21504])
        l1024_21 = self.cnn1024(x[:,:,21504:22528])
        l1024_22 = self.cnn1024(x[:,:,22528:23552])
        l1024_23 = self.cnn1024(x[:,:,23552:24576])
        l1024_24 = self.cnn1024(x[:,:,24576:25600])
        l1024_25 = self.cnn1024(x[:,:,25600:26624])
        l1024_26 = self.cnn1024(x[:,:,26624:27648])
        l1024_27 = self.cnn1024(x[:,:,27648:28672])
        l1024_28 = self.cnn1024(x[:,:,28672:29696])
        l1024_29 = self.cnn1024(x[:,:,29696:30720])
        l1024_30 = self.cnn1024(x[:,:,30720:31744])
        l1024_31 = self.cnn1024(x[:,:,31744:])

        out = torch.cat(( \
            l1024_0, l1024_1, l1024_2, l1024_3, l1024_4, l1024_5, l1024_6, l1024_7, l1024_8, l1024_9, l1024_10, l1024_11, l1024_12, l1024_13, l1024_14, l1024_15, l1024_16, l1024_17, l1024_18, l1024_19, l1024_20, l1024_21, l1024_22, l1024_23, l1024_24, l1024_25, l1024_26, l1024_27, l1024_28, l1024_29, l1024_30, l1024_31,\
            l2048_0, l2048_1, l2048_2, l2048_3, l2048_4, l2048_5, l2048_6, l2048_7, l2048_8, l2048_9, l2048_10, l2048_11, l2048_12, l2048_13, l2048_14, l2048_15,  \
            l4096_0, l4096_1, l4096_2, l4096_3, l4096_4, l4096_5, l4096_6, l4096_7,  \
            l8192_0, l8192_1, l8192_2, l8192_3), dim=1)

        return self.endDense(out)


class DiscriminatorMultiNetNo1024(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn8192 = nn.Sequential(
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

			nn.Linear(64*4, 64),
			nn.LeakyReLU(0.2),

			nn.Linear(64, 1)
		)

        self.cnn4096 = nn.Sequential(
			nn.Conv1d(1, 4, kernel_size = 8, stride = 4, padding = 0, bias = False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(4, 8, kernel_size = 5, stride = 3, padding = 0, bias = False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(8, 16, kernel_size = 5, stride = 2, padding = 0, bias = False),
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
            
            nn.Linear(64*9, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 1)
		)

        self.cnn2048 = nn.Sequential(
			nn.Conv1d(1, 4, kernel_size = 5, stride = 3, padding = 0, bias = False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(4, 8, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(8, 16, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(16, 16, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(16, 32, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(32, 64, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Flatten(),
            
            nn.Linear(64*20, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 1)
		)

        self.endDense = nn.Sequential(
			nn.Linear(28, 1),
			nn.Sigmoid()
		)
		
    def forward(self, x):

        l8192_0 = self.cnn8192(x[:,:,:8192])
        l8192_1 = self.cnn8192(x[:,:,8192:16384])
        l8192_2 = self.cnn8192(x[:,:,16384:24576])
        l8192_3 = self.cnn8192(x[:,:,24576:])


        l4096_0 = self.cnn4096(x[:,:,:4096])
        l4096_1 = self.cnn4096(x[:,:,4096:8192])
        l4096_2 = self.cnn4096(x[:,:,8192:12288])
        l4096_3 = self.cnn4096(x[:,:,12288:16384])
        l4096_4 = self.cnn4096(x[:,:,16384:20480])
        l4096_5 = self.cnn4096(x[:,:,20480:24576])
        l4096_6 = self.cnn4096(x[:,:,24576:28672])
        l4096_7 = self.cnn4096(x[:,:,28672:])


        l2048_0 = self.cnn2048(x[:,:,:2048])
        l2048_1 = self.cnn2048(x[:,:,2048:4096])
        l2048_2 = self.cnn2048(x[:,:,4096:6144])
        l2048_3 = self.cnn2048(x[:,:,6144:8192])
        l2048_4 = self.cnn2048(x[:,:,8192:10240])
        l2048_5 = self.cnn2048(x[:,:,10240:12288])
        l2048_6 = self.cnn2048(x[:,:,12288:14336])
        l2048_7 = self.cnn2048(x[:,:,14336:16384])
        l2048_8 = self.cnn2048(x[:,:,16384:18432])
        l2048_9 = self.cnn2048(x[:,:,18432:20480])
        l2048_10 = self.cnn2048(x[:,:,20480:22528])
        l2048_11 = self.cnn2048(x[:,:,22528:24576])
        l2048_12 = self.cnn2048(x[:,:,24576:26624])
        l2048_13 = self.cnn2048(x[:,:,26624:28672])
        l2048_14 = self.cnn2048(x[:,:,28672:30720])
        l2048_15 = self.cnn2048(x[:,:,30720:])

        out = torch.cat(( \
            l2048_0, l2048_1, l2048_2, l2048_3, l2048_4, l2048_5, l2048_6, l2048_7, l2048_8, l2048_9, l2048_10, l2048_11, l2048_12, l2048_13, l2048_14, l2048_15,  \
            l4096_0, l4096_1, l4096_2, l4096_3, l4096_4, l4096_5, l4096_6, l4096_7,  \
            l8192_0, l8192_1, l8192_2, l8192_3), dim=1)

        return self.endDense(out)


class DiscriminatorMultiNetWeightedAvg(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn8192 = nn.Sequential(
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

			nn.Linear(64*4, 64),
			nn.LeakyReLU(0.2),

			nn.Linear(64, 1)
		)

        self.cnn4096 = nn.Sequential(
			nn.Conv1d(1, 4, kernel_size = 8, stride = 4, padding = 0, bias = False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(4, 8, kernel_size = 5, stride = 3, padding = 0, bias = False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(8, 16, kernel_size = 5, stride = 2, padding = 0, bias = False),
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
            
            nn.Linear(64*9, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 1)
		)

        self.cnn2048 = nn.Sequential(
			nn.Conv1d(1, 4, kernel_size = 5, stride = 3, padding = 0, bias = False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(4, 8, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(8, 16, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(16, 16, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(16, 32, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(32, 64, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Flatten(),
            
            nn.Linear(64*20, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 1)
		)

        self.cnn1024 = nn.Sequential(
			nn.Conv1d(1, 4, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(4, 8, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(8, 16, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(16, 32, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(32, 32, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Flatten(),
            
            nn.Linear(32*30, 32),
            nn.LeakyReLU(0.2),

            nn.Linear(32, 1)
		)

        self.cnn512 = nn.Sequential(
			nn.Conv1d(1, 4, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(4, 8, kernel_size = 5, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(8, 16, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(16, 32, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(32, 32, kernel_size = 3, stride = 2, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Flatten(),
            
            nn.Linear(32*14, 16),
            nn.LeakyReLU(0.2),

            nn.Linear(16, 1)
		)

        self.endDense = nn.Sequential(
			nn.Linear(124, 1),
			nn.Sigmoid()
		)
		
    def forward(self, x):

        l8192_0 = self.cnn8192(x[:,:,:8192])
        l8192_1 = self.cnn8192(x[:,:,8192:16384])
        l8192_2 = self.cnn8192(x[:,:,16384:24576])
        l8192_3 = self.cnn8192(x[:,:,24576:])


        l4096_0 = self.cnn4096(x[:,:,:4096])
        l4096_1 = self.cnn4096(x[:,:,4096:8192])
        l4096_2 = self.cnn4096(x[:,:,8192:12288])
        l4096_3 = self.cnn4096(x[:,:,12288:16384])
        l4096_4 = self.cnn4096(x[:,:,16384:20480])
        l4096_5 = self.cnn4096(x[:,:,20480:24576])
        l4096_6 = self.cnn4096(x[:,:,24576:28672])
        l4096_7 = self.cnn4096(x[:,:,28672:])


        l2048_0 = self.cnn2048(x[:,:,:2048])
        l2048_1 = self.cnn2048(x[:,:,2048:4096])
        l2048_2 = self.cnn2048(x[:,:,4096:6144])
        l2048_3 = self.cnn2048(x[:,:,6144:8192])
        l2048_4 = self.cnn2048(x[:,:,8192:10240])
        l2048_5 = self.cnn2048(x[:,:,10240:12288])
        l2048_6 = self.cnn2048(x[:,:,12288:14336])
        l2048_7 = self.cnn2048(x[:,:,14336:16384])
        l2048_8 = self.cnn2048(x[:,:,16384:18432])
        l2048_9 = self.cnn2048(x[:,:,18432:20480])
        l2048_10 = self.cnn2048(x[:,:,20480:22528])
        l2048_11 = self.cnn2048(x[:,:,22528:24576])
        l2048_12 = self.cnn2048(x[:,:,24576:26624])
        l2048_13 = self.cnn2048(x[:,:,26624:28672])
        l2048_14 = self.cnn2048(x[:,:,28672:30720])
        l2048_15 = self.cnn2048(x[:,:,30720:])


        l1024_0 = self.cnn1024(x[:,:,:1024])
        l1024_1 = self.cnn1024(x[:,:,1024:2048])
        l1024_2 = self.cnn1024(x[:,:,2048:3072])
        l1024_3 = self.cnn1024(x[:,:,3072:4096])
        l1024_4 = self.cnn1024(x[:,:,4096:5120])
        l1024_5 = self.cnn1024(x[:,:,5120:6144])
        l1024_6 = self.cnn1024(x[:,:,6144:7168])
        l1024_7 = self.cnn1024(x[:,:,7168:8192])
        l1024_8 = self.cnn1024(x[:,:,8192:9216])
        l1024_9 = self.cnn1024(x[:,:,9216:10240])
        l1024_10 = self.cnn1024(x[:,:,10240:11264])
        l1024_11 = self.cnn1024(x[:,:,11264:12288])
        l1024_12 = self.cnn1024(x[:,:,12288:13312])
        l1024_13 = self.cnn1024(x[:,:,13312:14336])
        l1024_14 = self.cnn1024(x[:,:,14336:15360])
        l1024_15 = self.cnn1024(x[:,:,15360:16384])
        l1024_16 = self.cnn1024(x[:,:,16384:17408])
        l1024_17 = self.cnn1024(x[:,:,17408:18432])
        l1024_18 = self.cnn1024(x[:,:,18432:19456])
        l1024_19 = self.cnn1024(x[:,:,19456:20480])
        l1024_20 = self.cnn1024(x[:,:,20480:21504])
        l1024_21 = self.cnn1024(x[:,:,21504:22528])
        l1024_22 = self.cnn1024(x[:,:,22528:23552])
        l1024_23 = self.cnn1024(x[:,:,23552:24576])
        l1024_24 = self.cnn1024(x[:,:,24576:25600])
        l1024_25 = self.cnn1024(x[:,:,25600:26624])
        l1024_26 = self.cnn1024(x[:,:,26624:27648])
        l1024_27 = self.cnn1024(x[:,:,27648:28672])
        l1024_28 = self.cnn1024(x[:,:,28672:29696])
        l1024_29 = self.cnn1024(x[:,:,29696:30720])
        l1024_30 = self.cnn1024(x[:,:,30720:31744])
        l1024_31 = self.cnn1024(x[:,:,31744:])


        l512_0 = self.cnn512(x[:,:,:512])
        l512_1 = self.cnn512(x[:,:,512:1024])
        l512_2 = self.cnn512(x[:,:,1024:1536])
        l512_3 = self.cnn512(x[:,:,1536:2048])
        l512_4 = self.cnn512(x[:,:,2048:2560])
        l512_5 = self.cnn512(x[:,:,2560:3072])
        l512_6 = self.cnn512(x[:,:,3072:3584])
        l512_7 = self.cnn512(x[:,:,3584:4096])
        l512_8 = self.cnn512(x[:,:,4096:4608])
        l512_9 = self.cnn512(x[:,:,4608:5120])
        l512_10 = self.cnn512(x[:,:,5120:5632])
        l512_11 = self.cnn512(x[:,:,5632:6144])
        l512_12 = self.cnn512(x[:,:,6144:6656])
        l512_13 = self.cnn512(x[:,:,6656:7168])
        l512_14 = self.cnn512(x[:,:,7168:7680])
        l512_15 = self.cnn512(x[:,:,7680:8192])
        l512_16 = self.cnn512(x[:,:,8192:8704])
        l512_17 = self.cnn512(x[:,:,8704:9216])
        l512_18 = self.cnn512(x[:,:,9216:9728])
        l512_19 = self.cnn512(x[:,:,9728:10240])
        l512_20 = self.cnn512(x[:,:,10240:10752])
        l512_21 = self.cnn512(x[:,:,10752:11264])
        l512_22 = self.cnn512(x[:,:,11264:11776])
        l512_23 = self.cnn512(x[:,:,11776:12288])
        l512_24 = self.cnn512(x[:,:,12288:12800])
        l512_25 = self.cnn512(x[:,:,12800:13312])
        l512_26 = self.cnn512(x[:,:,13312:13824])
        l512_27 = self.cnn512(x[:,:,13824:14336])
        l512_28 = self.cnn512(x[:,:,14336:14848])
        l512_29 = self.cnn512(x[:,:,14848:15360])
        l512_30 = self.cnn512(x[:,:,15360:15872])
        l512_31 = self.cnn512(x[:,:,15872:16384])
        l512_32 = self.cnn512(x[:,:,16384:16896])
        l512_33 = self.cnn512(x[:,:,16896:17408])
        l512_34 = self.cnn512(x[:,:,17408:17920])
        l512_35 = self.cnn512(x[:,:,17920:18432])
        l512_36 = self.cnn512(x[:,:,18432:18944])
        l512_37 = self.cnn512(x[:,:,18944:19456])
        l512_38 = self.cnn512(x[:,:,19456:19968])
        l512_39 = self.cnn512(x[:,:,19968:20480])
        l512_40 = self.cnn512(x[:,:,20480:20992])
        l512_41 = self.cnn512(x[:,:,20992:21504])
        l512_42 = self.cnn512(x[:,:,21504:22016])
        l512_43 = self.cnn512(x[:,:,22016:22528])
        l512_44 = self.cnn512(x[:,:,22528:23040])
        l512_45 = self.cnn512(x[:,:,23040:23552])
        l512_46 = self.cnn512(x[:,:,23552:24064])
        l512_47 = self.cnn512(x[:,:,24064:24576])
        l512_48 = self.cnn512(x[:,:,24576:25088])
        l512_49 = self.cnn512(x[:,:,25088:25600])
        l512_50 = self.cnn512(x[:,:,25600:26112])
        l512_51 = self.cnn512(x[:,:,26112:26624])
        l512_52 = self.cnn512(x[:,:,26624:27136])
        l512_53 = self.cnn512(x[:,:,27136:27648])
        l512_54 = self.cnn512(x[:,:,27648:28160])
        l512_55 = self.cnn512(x[:,:,28160:28672])
        l512_56 = self.cnn512(x[:,:,28672:29184])
        l512_57 = self.cnn512(x[:,:,29184:29696])
        l512_58 = self.cnn512(x[:,:,29696:30208])
        l512_59 = self.cnn512(x[:,:,30208:30720])
        l512_60 = self.cnn512(x[:,:,30720:31232])
        l512_61 = self.cnn512(x[:,:,31232:31744])
        l512_62 = self.cnn512(x[:,:,31744:32256])
        l512_63 = self.cnn512(x[:,:,32256:])


        out = (((l512_0 + l512_1 + l512_2 + l512_3 + l512_4 + l512_5 + l512_6 + l512_7 + l512_8 + l512_9 + l512_10 + l512_11 + l512_12 + l512_13 + l512_14 + l512_15 + l512_16 + l512_17 + l512_18 + l512_19 + l512_20 + l512_21 + l512_22 + l512_23 + l512_24 + l512_25 + l512_26 + l512_27 + l512_28 + l512_29 + l512_30 + l512_31 + l512_32 + l512_33 + l512_34 + l512_35 + l512_36 + l512_37 + l512_38 + l512_39 + l512_40 + l512_41 + l512_42 + l512_43 + l512_44 + l512_45 + l512_46 + l512_47 + l512_48 + l512_49 + l512_50 + l512_51 + l512_52 + l512_53 + l512_54 + l512_55 + l512_56 + l512_57 + l512_58 + l512_59 + l512_60 + l512_61 + l512_62 + l512_63) / 64) +
            ((l1024_0 + l1024_1 + l1024_2 + l1024_3 + l1024_4 + l1024_5 + l1024_6 + l1024_7 + l1024_8 + l1024_9 + l1024_10 + l1024_11 + l1024_12 + l1024_13 + l1024_14 + l1024_15 + l1024_16 + l1024_17 + l1024_18 + l1024_19 + l1024_20 + l1024_21 + l1024_22 + l1024_23 + l1024_24 + l1024_25 + l1024_26 + l1024_27 + l1024_28 + l1024_29 + l1024_30 + l1024_31) / 32 ) +
            ((l2048_0 + l2048_1 + l2048_2 + l2048_3 + l2048_4 + l2048_5 + l2048_6 + l2048_7 + l2048_8 + l2048_9 + l2048_10 + l2048_11 + l2048_12 + l2048_13 + l2048_14 + l2048_15) / 16) + 
            ((l4096_0 + l4096_1 + l4096_2 + l4096_3 + l4096_4 + l4096_5 + l4096_6 + l4096_7) / 8) + 
            ((l8192_0 + l8192_1 + l8192_2 + l8192_3) / 4))

        return out # self.endDense(out)

# Original Discriminator
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
