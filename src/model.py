import torch.nn as nn

class Denoiser(nn.Module):
	def __init__(self, depth=10, conv_features=32):
		super(Denoiser, self).__init__()
		in_layer = nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
								 nn.ReLU(inplace=True))
		
		inter_layer = nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
									nn.BatchNorm2d(num_features),
									nn.ReLU(inplace=True))
		
		out_layer = nn.Conv2d(num_features, 3, kernel_size=3, padding=1)

		self.layers = [in_layer]
		self.layers += [inter_layer]*depth
		self.layers += out_layer

	def forward(self, x):
		y = x
		noise = self.layers(x)
		return y - noise