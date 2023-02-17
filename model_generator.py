from torch import nn

### Model version 3
class CNNGenerator(nn.Module):
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
		self.cnn4 = nn.Sequential(
                 	nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(64),
                	nn.ReLU(True),
                )
		self.cnn8 = nn.Sequential(
                 	nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(128),
                	nn.ReLU(True),
                )
		self.cnn16 = nn.Sequential(
                 	nn.Conv1d(128, 256, kernel_size=16, stride=1, padding=0, bias=False),
                	nn.BatchNorm1d(256),
                	nn.ReLU(True),
                )
		self.cnn32 = nn.Sequential(
                 	nn.Conv1d(256, 256, kernel_size=32, stride=1, padding=0, bias=False),
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

		# print(1, out.shape)
		out = self.avgPool2(out)
		# print(2, out.shape)
		res2 = self.cnn2(out)
		out = res2

		# print(3, out.shape)
		out = self.avgPool2(out)
		# print(4, out.shape)
		res4 = self.cnn4(out)
		out = res4

		# print(5, out.shape)
		out = self.avgPool2(out)
		# print(6, out.shape)
		res8 = self.cnn8(out)
		out = res8

		# print(7, out.shape)
		out = self.avgPool2(out)
		# print(8, out.shape)
		res16 = self.cnn16(out)
		out = res16

		# print(9, out.shape)
		out = self.avgPool2(out)
		# print(10, out.shape)
		res32 = self.cnn32(out)
		out = res32

		# print(11, out.shape)
		out = self.avgPool2(out)
		# print(12, out.shape)
		out = self.cnn64(out)
		# print(13, out.shape)

		out = self.bridge1(out)
		# print(14, out.shape)
		out = self.bridge2(out)
		# print(15, out.shape)
		out = self.bridge3(out)
		# print(16, out.shape)

		out = self.cnnTrans64(out)
		# print(17, out.shape)
		out = self.upsample2(out)
		# print(18, out.shape)
		out = out + res32

		out = self.cnnTrans32(out)
		# print(19, out.shape)
		out = self.upsample2(out)
		# print(20, out[:,:,0:-1].shape)
		# print(20, res16.shape)
		out = out[:,:,0:-1] + res16

		out = self.cnnTrans16(out)
		# print(21, out.shape)
		out = self.upsample2(out)
		# print(22, out.shape)
		out = out + res8

		out = self.cnnTrans8(out)
		# print(23, out.shape)
		out = self.upsample2(out)
		# print(24, out.shape)
		out = out[:,:,0:-1] + res4

		out = self.cnnTrans4(out)
		# print(25, out.shape)
		out = self.upsample2(out)
		# print(26, out.shape)
		out = out[:,:,0:-1] + res2

		out = self.cnnTrans2(out)
		# print(27, out.shape)
		out = self.upsample2(out)
		# print(28, out.shape)
		out = out + res1

		out = self.cnnTransOut(out)
		# print(29, out.shape)
		return out
