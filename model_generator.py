from torch import nn
import torch


### Model big version 1 double cnn when downsampling
class CNNGeneratorBig1(nn.Module):
	def __init__(self):
		super().__init__()
		self.avgPool2 = nn.AvgPool1d(2, ceil_mode=True)
		self.upsample2 = nn.Upsample(scale_factor=2, mode='linear')
		self.cnn1 = nn.Sequential(
                 	nn.Conv1d(1, 16, kernel_size=1, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(16),
                	nn.ReLU(True),
                )
		self.cnn2 = nn.Sequential(
                 	nn.Conv1d(16, 32, kernel_size=2, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(32),
                	nn.ReLU(True),
                )
		self.cnn22 = nn.Sequential(
                 	nn.Conv1d(32, 32, kernel_size=2, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(32),
                	nn.ReLU(True),
                )
		self.cnn4 = nn.Sequential(
                 	nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(64),
                	nn.ReLU(True),
                )
		self.cnn44 = nn.Sequential(
                 	nn.Conv1d(64, 64, kernel_size=4, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(64),
                	nn.ReLU(True),
                )
		self.cnn8 = nn.Sequential(
                 	nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(128),
                	nn.ReLU(True),
                )
		self.cnn88 = nn.Sequential(
                 	nn.Conv1d(128, 128, kernel_size=8, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(128),
                	nn.ReLU(True),
                )
		self.cnn16 = nn.Sequential(
                 	nn.Conv1d(128, 256, kernel_size=16, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn1616 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=16, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn32 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn3232 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn64 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=64, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.bridge1 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.bridge2 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.bridge3 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnnTrans64 = nn.Sequential(
                 	nn.ConvTranspose1d(256, 256, kernel_size=64, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnnTrans32 = nn.Sequential(
                 	nn.ConvTranspose1d(256, 256, kernel_size=32, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnnTrans16 = nn.Sequential(
                 	nn.ConvTranspose1d(256, 128, kernel_size=16, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(128),
                	nn.ReLU(True),
                )
		self.cnnTrans8 = nn.Sequential(
                 	nn.ConvTranspose1d(128, 64, kernel_size=8, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(64),
                	nn.ReLU(True),
                )
		self.cnnTrans4 = nn.Sequential(
                 	nn.ConvTranspose1d(64, 32, kernel_size=4, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(32),
                	nn.ReLU(True),
                )
		self.cnnTrans2 = nn.Sequential(
                 	nn.ConvTranspose1d(32, 16, kernel_size=2, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(16),
                	nn.ReLU(True),
                )
		self.cnnTransOut = nn.Sequential(
             	nn.Conv1d(16, 1, kernel_size=1, stride=1, padding=0, bias=False),
            )

	def forward(self, z):
		res1 = self.cnn1(z)
		out = res1

		# # print(1, out.shape)
		out = self.avgPool2(out)
		# # print(2, out.shape)
		out = self.cnn2(out)
		# # print(3, out.shape)
		res22 = self.cnn22(out)
		out = res22

		# # print(4, out.shape)
		out = self.avgPool2(out)
		# # print(5, out.shape)
		out = self.cnn4(out)
		# # print(6, out.shape)
		res44 = self.cnn44(out)
		out = res44

		# # print(7, out.shape)
		out = self.avgPool2(out)
		# # print(8, out.shape)
		out = self.cnn8(out)
		# # print(9, out.shape)
		res88 = self.cnn88(out)
		out = res88

		# # print(10, out.shape)
		out = self.avgPool2(out)
		# # print(11, out.shape)
		out = self.cnn16(out)
		# # print(12, out.shape)
		res1616 = self.cnn1616(out)
		out = res1616

		# # print(13, out.shape)
		out = self.avgPool2(out)
		# # print(14, out.shape)
		out = self.cnn32(out)
		# # print(15, out.shape)
		res3232 = self.cnn3232(out)
		out = res3232

		# # print(16, out.shape)
		out = self.avgPool2(out)
		# # print(17, out.shape)
		out = self.cnn64(out)
		# # print(18, out.shape)

		out = self.bridge1(out)
		# # print(19, out.shape)
		out = self.bridge2(out)
		# # print(20, out.shape)
		out = self.bridge3(out)
		# # print(21, out.shape)

		out = self.cnnTrans64(out)
		# # print(22, out.shape)
		out = self.upsample2(out)
		# # print(23, out.shape)
		out = out + res3232

		out = self.cnnTrans32(out)
		# # print(24, out.shape)
		out = self.upsample2(out)
		# # print(25, out.shape)
		out = out[:,:,0:-1] + res1616

		out = self.cnnTrans16(out)
		# # print(26, out.shape)
		out = self.upsample2(out)
		# # print(27, out.shape)
		out = out + res88

		out = self.cnnTrans8(out)
		# # print(28, out.shape)
		out = self.upsample2(out)
		# # print(29, out.shape)
		out = out[:,:,0:-1] + res44

		out = self.cnnTrans4(out)
		# # print(30, out.shape)
		out = self.upsample2(out)
		# # print(31, out.shape)
		out = out[:,:,0:-1] + res22

		out = self.cnnTrans2(out)
		# # print(32, out.shape)
		out = self.upsample2(out)
		# # print(33, out.shape)
		out = out + res1

		out = self.cnnTransOut(out)
		# # print(34, out.shape)
		return out


### Model big version 2 double cnn after upscaling and when downsampling
class CNNGeneratorBig2(nn.Module):
	def __init__(self):
		super().__init__()
		self.avgPool2 = nn.AvgPool1d(2, ceil_mode=True)
		self.upsample2 = nn.Upsample(scale_factor=2, mode='linear')
		self.cnn1 = nn.Sequential(
                 	nn.Conv1d(1, 16, kernel_size=1, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(16),
                	nn.ReLU(True),
                )
		self.cnn2 = nn.Sequential(
                 	nn.Conv1d(16, 32, kernel_size=2, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(32),
                	nn.ReLU(True),
                )
		self.cnn22 = nn.Sequential(
                 	nn.Conv1d(32, 32, kernel_size=2, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(32),
                	nn.ReLU(True),
                )
		self.cnn4 = nn.Sequential(
                 	nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(64),
                	nn.ReLU(True),
                )
		self.cnn44 = nn.Sequential(
                 	nn.Conv1d(64, 64, kernel_size=4, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(64),
                	nn.ReLU(True),
                )
		self.cnn8 = nn.Sequential(
                 	nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(128),
                	nn.ReLU(True),
                )
		self.cnn88 = nn.Sequential(
                 	nn.Conv1d(128, 128, kernel_size=8, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(128),
                	nn.ReLU(True),
                )
		self.cnn16 = nn.Sequential(
                 	nn.Conv1d(128, 256, kernel_size=16, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn1616 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=16, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn32 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn3232 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn64 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=64, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.bridge1 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.bridge2 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.bridge3 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnnTrans64 = nn.Sequential(
                 	nn.ConvTranspose1d(256, 256, kernel_size=64, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnnTrans32 = nn.Sequential(
                 	nn.ConvTranspose1d(256, 256, kernel_size=32, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnnTrans16 = nn.Sequential(
                 	nn.ConvTranspose1d(256, 128, kernel_size=16, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(128),
                	nn.ReLU(True),
                )
		self.cnnTrans8 = nn.Sequential(
                 	nn.ConvTranspose1d(128, 64, kernel_size=8, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(64),
                	nn.ReLU(True),
                )
		self.cnnTrans4 = nn.Sequential(
                 	nn.ConvTranspose1d(64, 32, kernel_size=4, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(32),
                	nn.ReLU(True),
                )
		self.cnnTrans2 = nn.Sequential(
                 	nn.ConvTranspose1d(32, 16, kernel_size=2, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(16),
                	nn.ReLU(True),
                )
		self.cnnTransOut = nn.Sequential(
             	nn.Conv1d(16, 1, kernel_size=1, stride=1, padding=0, bias=False),
            )

	def forward(self, z):
		res1 = self.cnn1(z)
		out = res1

		# # print(1, out.shape)
		out = self.avgPool2(out)
		# # print(2, out.shape)
		out = self.cnn2(out)
		# # print(3, out.shape)
		res22 = self.cnn22(out)
		out = res22

		# # print(4, out.shape)
		out = self.avgPool2(out)
		# # print(5, out.shape)
		out = self.cnn4(out)
		# # print(6, out.shape)
		res44 = self.cnn44(out)
		out = res44

		# # print(7, out.shape)
		out = self.avgPool2(out)
		# # print(8, out.shape)
		out = self.cnn8(out)
		# # print(9, out.shape)
		res88 = self.cnn88(out)
		out = res88

		# # print(10, out.shape)
		out = self.avgPool2(out)
		# # print(11, out.shape)
		out = self.cnn16(out)
		# # print(12, out.shape)
		res1616 = self.cnn1616(out)
		out = res1616

		# # print(13, out.shape)
		out = self.avgPool2(out)
		# # print(14, out.shape)
		out = self.cnn32(out)
		# # print(15, out.shape)
		res3232 = self.cnn3232(out)
		out = res3232

		# # print(16, out.shape)
		out = self.avgPool2(out)
		# # print(17, out.shape)
		out = self.cnn64(out)
		# # print(18, out.shape)

		out = self.bridge1(out)
		# # print(19, out.shape)
		out = self.bridge2(out)
		# # print(20, out.shape)
		out = self.bridge3(out)
		# # print(21, out.shape)

		out = self.cnnTrans64(out)
		# # print(22, out.shape)
		out = self.upsample2(out)
		# # print(23, out.shape)
		out = out + res3232

		out = self.cnnTrans32(out)
		out = self.cnn3232(out)
		# # print(24, out.shape)
		out = self.upsample2(out)
		# # print(25, out.shape)
		out = out[:,:,0:-1] + res1616

		out = self.cnnTrans16(out)
		out = self.cnn88(out)
		# # print(26, out.shape)
		out = self.upsample2(out)
		# # print(27, out.shape)
		out = out + res88

		out = self.cnnTrans8(out)
		out = self.cnn44(out)
		# # print(28, out.shape)
		out = self.upsample2(out)
		# # print(29, out.shape)
		out = out[:,:,0:-1] + res44

		out = self.cnnTrans4(out)
		out = self.cnn22(out)
		# # print(30, out.shape)
		out = self.upsample2(out)
		# # print(31, out.shape)
		out = out[:,:,0:-1] + res22

		out = self.cnnTrans2(out)
		# # print(32, out.shape)
		out = self.upsample2(out)
		# # print(33, out.shape)
		out = out + res1

		out = self.cnnTransOut(out)
		# # print(34, out.shape)
		return out


### From big v2 with concatenation
class CNNGeneratorBigConcat(nn.Module):
	def __init__(self):
		super().__init__()
		self.avgPool2 = nn.AvgPool1d(2, ceil_mode=True)
		self.upsample2 = nn.Upsample(scale_factor=2, mode='linear')
		self.cnn1 = nn.Sequential(
                 	nn.Conv1d(1, 16, kernel_size=1, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(16),
                	nn.ReLU(True),
                )
		self.cnn2 = nn.Sequential(
                 	nn.Conv1d(16, 32, kernel_size=2, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(32),
                	nn.ReLU(True),
                )
		self.cnn2_ = nn.Sequential(
                 	nn.Conv1d(32, 32, kernel_size=2, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(32),
                	nn.ReLU(True),
                )
		self.cnn4 = nn.Sequential(
                 	nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(64),
                	nn.ReLU(True),
                )
		self.cnn4_ = nn.Sequential(
                 	nn.Conv1d(64, 64, kernel_size=4, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(64),
                	nn.ReLU(True),
                )
		self.cnn8 = nn.Sequential(
                 	nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(128),
                	nn.ReLU(True),
                )
		self.cnn8_ = nn.Sequential(
                 	nn.Conv1d(128, 128, kernel_size=8, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(128),
                	nn.ReLU(True),
                )
		self.cnn16 = nn.Sequential(
                 	nn.Conv1d(128, 256, kernel_size=16, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn16_ = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=16, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn32 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn32_ = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn64 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=64, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.bridge1 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.bridge2 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.bridge3 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnnTrans64 = nn.Sequential(
                 	nn.ConvTranspose1d(256, 256, kernel_size=64, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnnTrans32 = nn.Sequential(
                 	nn.ConvTranspose1d(512, 256, kernel_size=32, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnnTrans16 = nn.Sequential(
                 	nn.ConvTranspose1d(512, 128, kernel_size=16, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(128),
                	nn.ReLU(True),
                )
		self.cnnTrans8 = nn.Sequential(
                 	nn.ConvTranspose1d(256, 64, kernel_size=8, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(64),
                	nn.ReLU(True),
                )
		self.cnnTrans4 = nn.Sequential(
                 	nn.ConvTranspose1d(128, 32, kernel_size=4, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(32),
                	nn.ReLU(True),
                )
		self.cnnTrans2 = nn.Sequential(
                 	nn.ConvTranspose1d(64, 16, kernel_size=2, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(16),
                	nn.ReLU(True),
                )
		self.cnnTransOut = nn.Sequential(
             	nn.ConvTranspose1d(32, 1, kernel_size=1, stride=1, padding=0, bias=False),
            )

	def forward(self, z):
		res1 = self.cnn1(z)
		out = res1

		# print(1, out.shape)
		out = self.avgPool2(out)
		# print(2, out.shape)
		out = self.cnn2(out)
		# print(3, out.shape)
		res2 = self.cnn2_(out)
		out = res2

		# print(4, out.shape)
		out = self.avgPool2(out)
		# print(5, out.shape)
		out = self.cnn4(out)
		# print(6, out.shape)
		res4 = self.cnn4_(out)
		out = res4

		# print(7, out.shape)
		out = self.avgPool2(out)
		# print(8, out.shape)
		out = self.cnn8(out)
		# print(9, out.shape)
		res8 = self.cnn8_(out)
		out = res8

		# print(10, out.shape)
		out = self.avgPool2(out)
		# print(11, out.shape)
		out = self.cnn16(out)
		# print(12, out.shape)
		res16 = self.cnn16_(out)
		out = res16

		# print(13, out.shape)
		out = self.avgPool2(out)
		# print(14, out.shape)
		out = self.cnn32(out)
		# print(15, out.shape)
		res32 = self.cnn32_(out)
		out = res32

		# print(16, out.shape)
		out = self.avgPool2(out)
		# print(17, out.shape)
		out = self.cnn64(out)
		# print(18, out.shape)

		out = self.bridge1(out)
		# print(19, out.shape)
		out = self.bridge2(out)
		# print(20, out.shape)
		out = self.bridge3(out)
		# print(21, out.shape)

		out = self.cnnTrans64(out)
		# print(22, out.shape)
		out = self.upsample2(out)
		# print(23, out.shape)
		out = torch.cat((out, res32), dim=1) 

		out = self.cnnTrans32(out)
		# print(24, out.shape)
		out = self.upsample2(out)
		# print(25, out.shape)
		out = torch.cat((out[:,:,0:-1], res16), dim=1) 

		out = self.cnnTrans16(out)
		# print(26, out.shape)
		out = self.upsample2(out)
		# print(27, out.shape)
		out = torch.cat((out, res8), dim=1) 

		out = self.cnnTrans8(out)
		# print(28, out.shape)
		out = self.upsample2(out)
		# print(29, out.shape)
		out = torch.cat((out[:,:,0:-1], res4), dim=1) 

		out = self.cnnTrans4(out)
		# print(30, out.shape)
		out = self.upsample2(out)
		# print(31, out.shape)
		out = torch.cat((out[:,:,0:-1], res2), dim=1) 

		out = self.cnnTrans2(out)
		# print(32, out.shape)
		out = self.upsample2(out)
		# print(33, out.shape)
		out = torch.cat((out, res1), dim=1)

		out = self.cnnTransOut(out)
		# print(34, out.shape)
		return out


### From big concatv2 no RES1 but has res1
class CNNGeneratorBCNores1(nn.Module):
	def __init__(self):
		super().__init__()
		self.avgPool2 = nn.AvgPool1d(2, ceil_mode=True)
		self.upsample2 = nn.Upsample(scale_factor=2, mode='linear')
		self.cnn1 = nn.Sequential(
                 	nn.Conv1d(1, 16, kernel_size=1, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(16),
                	nn.ReLU(True),
                )
		self.cnn2 = nn.Sequential(
                 	nn.Conv1d(16, 32, kernel_size=2, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(32),
                	nn.ReLU(True),
                )
		self.cnn2_ = nn.Sequential(
                 	nn.Conv1d(32, 32, kernel_size=2, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(32),
                	nn.ReLU(True),
                )
		self.cnn4 = nn.Sequential(
                 	nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(64),
                	nn.ReLU(True),
                )
		self.cnn4_ = nn.Sequential(
                 	nn.Conv1d(64, 64, kernel_size=4, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(64),
                	nn.ReLU(True),
                )
		self.cnn8 = nn.Sequential(
                 	nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(128),
                	nn.ReLU(True),
                )
		self.cnn8_ = nn.Sequential(
                 	nn.Conv1d(128, 128, kernel_size=8, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(128),
                	nn.ReLU(True),
                )
		self.cnn16 = nn.Sequential(
                 	nn.Conv1d(128, 256, kernel_size=16, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn16_ = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=16, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn32 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn32_ = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn64 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=64, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.bridge1 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.bridge2 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.bridge3 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnnTrans64 = nn.Sequential(
                 	nn.ConvTranspose1d(256, 256, kernel_size=64, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnnTrans32 = nn.Sequential(
                 	nn.ConvTranspose1d(512, 256, kernel_size=32, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnnTrans16 = nn.Sequential(
                 	nn.ConvTranspose1d(512, 128, kernel_size=16, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(128),
                	nn.ReLU(True),
                )
		self.cnnTrans8 = nn.Sequential(
                 	nn.ConvTranspose1d(256, 64, kernel_size=8, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(64),
                	nn.ReLU(True),
                )
		self.cnnTrans4 = nn.Sequential(
                 	nn.ConvTranspose1d(128, 32, kernel_size=4, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(32),
                	nn.ReLU(True),
                )
		self.cnnTrans2 = nn.Sequential(
                 	nn.ConvTranspose1d(64, 16, kernel_size=2, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(16),
                	nn.ReLU(True),
                )
		self.cnnTransOut = nn.Sequential(
             	nn.ConvTranspose1d(16, 1, kernel_size=1, stride=1, padding=0, bias=False),
            )

	def forward(self, z):
		# res1 = self.cnn1(z)
		# out = res1
		out = self.cnn1(z)
		out = out

		# print(1, out.shape)
		out = self.avgPool2(out)
		# print(2, out.shape)
		out = self.cnn2(out)
		# print(3, out.shape)
		res2 = self.cnn2_(out)
		out = res2

		# print(4, out.shape)
		out = self.avgPool2(out)
		# print(5, out.shape)
		out = self.cnn4(out)
		# print(6, out.shape)
		res4 = self.cnn4_(out)
		out = res4

		# print(7, out.shape)
		out = self.avgPool2(out)
		# print(8, out.shape)
		out = self.cnn8(out)
		# print(9, out.shape)
		res8 = self.cnn8_(out)
		out = res8

		# print(10, out.shape)
		out = self.avgPool2(out)
		# print(11, out.shape)
		out = self.cnn16(out)
		# print(12, out.shape)
		res16 = self.cnn16_(out)
		out = res16

		# print(13, out.shape)
		out = self.avgPool2(out)
		# print(14, out.shape)
		out = self.cnn32(out)
		# print(15, out.shape)
		res32 = self.cnn32_(out)
		out = res32

		# print(16, out.shape)
		out = self.avgPool2(out)
		# print(17, out.shape)
		out = self.cnn64(out)
		# print(18, out.shape)

		out = self.bridge1(out)
		# print(19, out.shape)
		out = self.bridge2(out)
		# print(20, out.shape)
		out = self.bridge3(out)
		# print(21, out.shape)

		out = self.cnnTrans64(out)
		# print(22, out.shape)
		out = self.upsample2(out)
		# print(23, out.shape)
		out = torch.cat((out, res32), dim=1) 

		out = self.cnnTrans32(out)
		# print(24, out.shape)
		out = self.upsample2(out)
		# print(25, out.shape)
		out = torch.cat((out[:,:,0:-1], res16), dim=1) 

		out = self.cnnTrans16(out)
		# print(26, out.shape)
		out = self.upsample2(out)
		# print(27, out.shape)
		out = torch.cat((out, res8), dim=1) 

		out = self.cnnTrans8(out)
		# print(28, out.shape)
		out = self.upsample2(out)
		# print(29, out.shape)
		out = torch.cat((out[:,:,0:-1], res4), dim=1) 

		out = self.cnnTrans4(out)
		# print(30, out.shape)
		out = self.upsample2(out)
		# print(31, out.shape)
		out = torch.cat((out[:,:,0:-1], res2), dim=1) 

		out = self.cnnTrans2(out)
		# print(32, out.shape)
		out = self.upsample2(out)
		# print(33, out.shape)
		#out = torch.cat((out, res1), dim=1)

		out = self.cnnTransOut(out)
		# print(34, out.shape)
		return out
	

### From big concatv2 no CNN1
class CNNGeneratorBCNocnn1(nn.Module):
	def __init__(self):
		super().__init__()
		self.avgPool2 = nn.AvgPool1d(2, ceil_mode=True)
		self.upsample2 = nn.Upsample(scale_factor=2, mode='linear')
		self.cnn1 = nn.Sequential(
                 	nn.Conv1d(1, 16, kernel_size=1, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(16),
                	nn.ReLU(True),
                )
		self.cnn2 = nn.Sequential(
                 	nn.Conv1d(1, 32, kernel_size=2, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(32),
                	nn.ReLU(True),
                )
		self.cnn2_ = nn.Sequential(
                 	nn.Conv1d(32, 32, kernel_size=2, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(32),
                	nn.ReLU(True),
                )
		self.cnn4 = nn.Sequential(
                 	nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(64),
                	nn.ReLU(True),
                )
		self.cnn4_ = nn.Sequential(
                 	nn.Conv1d(64, 64, kernel_size=4, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(64),
                	nn.ReLU(True),
                )
		self.cnn8 = nn.Sequential(
                 	nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(128),
                	nn.ReLU(True),
                )
		self.cnn8_ = nn.Sequential(
                 	nn.Conv1d(128, 128, kernel_size=8, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(128),
                	nn.ReLU(True),
                )
		self.cnn16 = nn.Sequential(
                 	nn.Conv1d(128, 256, kernel_size=16, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn16_ = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=16, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn32 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn32_ = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn64 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=64, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.bridge1 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.bridge2 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.bridge3 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnnTrans64 = nn.Sequential(
                 	nn.ConvTranspose1d(256, 256, kernel_size=64, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnnTrans32 = nn.Sequential(
                 	nn.ConvTranspose1d(512, 256, kernel_size=32, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnnTrans16 = nn.Sequential(
                 	nn.ConvTranspose1d(512, 128, kernel_size=16, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(128),
                	nn.ReLU(True),
                )
		self.cnnTrans8 = nn.Sequential(
                 	nn.ConvTranspose1d(256, 64, kernel_size=8, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(64),
                	nn.ReLU(True),
                )
		self.cnnTrans4 = nn.Sequential(
                 	nn.ConvTranspose1d(128, 32, kernel_size=4, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(32),
                	nn.ReLU(True),
                )
		self.cnnTrans2 = nn.Sequential(
                 	nn.ConvTranspose1d(64, 16, kernel_size=2, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(16),
                	nn.ReLU(True),
                )
		self.cnnTransOut = nn.Sequential(
             	nn.ConvTranspose1d(16, 1, kernel_size=1, stride=1, padding=0, bias=False),
            )

	def forward(self, z):
		# res1 = self.cnn1(z)
		# out = res1

		# print(1, out.shape)
		out = self.avgPool2(z)
		# print(2, out.shape)
		out = self.cnn2(out)
		# print(3, out.shape)
		res2 = self.cnn2_(out)
		out = res2

		# print(4, out.shape)
		out = self.avgPool2(out)
		# print(5, out.shape)
		out = self.cnn4(out)
		# print(6, out.shape)
		res4 = self.cnn4_(out)
		out = res4

		# print(7, out.shape)
		out = self.avgPool2(out)
		# print(8, out.shape)
		out = self.cnn8(out)
		# print(9, out.shape)
		res8 = self.cnn8_(out)
		out = res8

		# print(10, out.shape)
		out = self.avgPool2(out)
		# print(11, out.shape)
		out = self.cnn16(out)
		# print(12, out.shape)
		res16 = self.cnn16_(out)
		out = res16

		# print(13, out.shape)
		out = self.avgPool2(out)
		# print(14, out.shape)
		out = self.cnn32(out)
		# print(15, out.shape)
		res32 = self.cnn32_(out)
		out = res32

		# print(16, out.shape)
		out = self.avgPool2(out)
		# print(17, out.shape)
		out = self.cnn64(out)
		# print(18, out.shape)

		out = self.bridge1(out)
		# print(19, out.shape)
		out = self.bridge2(out)
		# print(20, out.shape)
		out = self.bridge3(out)
		# print(21, out.shape)

		out = self.cnnTrans64(out)
		# print(22, out.shape)
		out = self.upsample2(out)
		# print(23, out.shape)
		out = torch.cat((out, res32), dim=1) 

		out = self.cnnTrans32(out)
		# print(24, out.shape)
		out = self.upsample2(out)
		# print(25, out.shape)
		out = torch.cat((out[:,:,0:-1], res16), dim=1) 

		out = self.cnnTrans16(out)
		# print(26, out.shape)
		out = self.upsample2(out)
		# print(27, out.shape)
		out = torch.cat((out, res8), dim=1) 

		out = self.cnnTrans8(out)
		# print(28, out.shape)
		out = self.upsample2(out)
		# print(29, out.shape)
		out = torch.cat((out[:,:,0:-1], res4), dim=1) 

		out = self.cnnTrans4(out)
		# print(30, out.shape)
		out = self.upsample2(out)
		# print(31, out.shape)
		out = torch.cat((out[:,:,0:-1], res2), dim=1) 

		out = self.cnnTrans2(out)
		# print(32, out.shape)
		out = self.upsample2(out)
		# print(33, out.shape)
		#out = torch.cat((out, res1), dim=1)

		out = self.cnnTransOut(out)
		# print(34, out.shape)
		return out

### From big concatv2 + more convolutions in the end
class CNNGeneratorBigConcatEndCnn(nn.Module):
	def __init__(self):
		super().__init__()
		self.avgPool2 = nn.AvgPool1d(2, ceil_mode=True)
		self.upsample2 = nn.Upsample(scale_factor=2, mode='linear')
		self.cnn1 = nn.Sequential(
                 	nn.Conv1d(1, 16, kernel_size=1, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(16),
                	nn.ReLU(True),
                )
		self.cnn2 = nn.Sequential(
                 	nn.Conv1d(16, 32, kernel_size=2, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(32),
                	nn.ReLU(True),
                )
		self.cnn2_ = nn.Sequential(
                 	nn.Conv1d(32, 32, kernel_size=2, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(32),
                	nn.ReLU(True),
                )
		self.cnn4 = nn.Sequential(
                 	nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(64),
                	nn.ReLU(True),
                )
		self.cnn4_ = nn.Sequential(
                 	nn.Conv1d(64, 64, kernel_size=4, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(64),
                	nn.ReLU(True),
                )
		self.cnn8 = nn.Sequential(
                 	nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(128),
                	nn.ReLU(True),
                )
		self.cnn8_ = nn.Sequential(
                 	nn.Conv1d(128, 128, kernel_size=8, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(128),
                	nn.ReLU(True),
                )
		self.cnn16 = nn.Sequential(
                 	nn.Conv1d(128, 256, kernel_size=16, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn16_ = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=16, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn32 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn32_ = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn64 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=64, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.bridge1 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.bridge2 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.bridge3 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding="same", bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnnTrans64 = nn.Sequential(
                 	nn.ConvTranspose1d(256, 256, kernel_size=64, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnnTrans32 = nn.Sequential(
                 	nn.ConvTranspose1d(512, 256, kernel_size=32, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnnTrans16 = nn.Sequential(
                 	nn.ConvTranspose1d(512, 128, kernel_size=16, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(128),
                	nn.ReLU(True),
                )
		self.cnnTrans8 = nn.Sequential(
                 	nn.ConvTranspose1d(256, 64, kernel_size=8, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(64),
                	nn.ReLU(True),
                )
		self.cnnTrans4 = nn.Sequential(
                 	nn.ConvTranspose1d(128, 32, kernel_size=4, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(32),
                	nn.ReLU(True),
                )
		self.cnnTrans2 = nn.Sequential(
                 	nn.ConvTranspose1d(64, 16, kernel_size=2, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(16),
                	nn.ReLU(True),
                )
		self.cnnTransOut = nn.Sequential(
             	nn.ConvTranspose1d(32, 4, kernel_size=1, stride=1, padding=0, bias=False),
            )
		self.cnnlast = nn.Sequential(
				nn.Conv1d(4, 1, kernel_size=3, stride=1, padding="same", bias=False),
			)

	def forward(self, z):
		res1 = self.cnn1(z)
		out = res1

		# print(1, out.shape)
		out = self.avgPool2(out)
		# print(2, out.shape)
		out = self.cnn2(out)
		# print(3, out.shape)
		res2 = self.cnn2_(out)
		out = res2

		# print(4, out.shape)
		out = self.avgPool2(out)
		# print(5, out.shape)
		out = self.cnn4(out)
		# print(6, out.shape)
		res4 = self.cnn4_(out)
		out = res4

		# print(7, out.shape)
		out = self.avgPool2(out)
		# print(8, out.shape)
		out = self.cnn8(out)
		# print(9, out.shape)
		res8 = self.cnn8_(out)
		out = res8

		# print(10, out.shape)
		out = self.avgPool2(out)
		# print(11, out.shape)
		out = self.cnn16(out)
		# print(12, out.shape)
		res16 = self.cnn16_(out)
		out = res16

		# print(13, out.shape)
		out = self.avgPool2(out)
		# print(14, out.shape)
		out = self.cnn32(out)
		# print(15, out.shape)
		res32 = self.cnn32_(out)
		out = res32

		# print(16, out.shape)
		out = self.avgPool2(out)
		# print(17, out.shape)
		out = self.cnn64(out)
		# print(18, out.shape)

		out = self.bridge1(out)
		# print(19, out.shape)
		out = self.bridge2(out)
		# print(20, out.shape)
		out = self.bridge3(out)
		# print(21, out.shape)

		out = self.cnnTrans64(out)
		# print(22, out.shape)
		out = self.upsample2(out)
		# print(23, out.shape)
		out = torch.cat((out, res32), dim=1) 

		out = self.cnnTrans32(out)
		# print(24, out.shape)
		out = self.upsample2(out)
		# print(25, out.shape)
		out = torch.cat((out[:,:,0:-1], res16), dim=1) 

		out = self.cnnTrans16(out)
		# print(26, out.shape)
		out = self.upsample2(out)
		# print(27, out.shape)
		out = torch.cat((out, res8), dim=1) 

		out = self.cnnTrans8(out)
		# print(28, out.shape)
		out = self.upsample2(out)
		# print(29, out.shape)
		out = torch.cat((out[:,:,0:-1], res4), dim=1) 

		out = self.cnnTrans4(out)
		# print(30, out.shape)
		out = self.upsample2(out)
		# print(31, out.shape)
		out = torch.cat((out[:,:,0:-1], res2), dim=1) 

		out = self.cnnTrans2(out)
		# print(32, out.shape)
		out = self.upsample2(out)
		# print(33, out.shape)
		out = torch.cat((out, res1), dim=1)

		out = self.cnnTransOut(out)
		out = self.cnnlast(out)
		# print(34, out.shape)
		return out


