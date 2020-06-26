import torch.nn as nn

class Denoiser(nn.Module):
	def __init__(self, depth=16, conv_features=64):
		super(Denoiser, self).__init__()
		in_layer = nn.Sequential(nn.Conv2d(3, depth, kernel_size=3, stride=1, padding=1),
								 nn.ReLU(inplace=True))
		
		inter_layer = nn.Sequential(nn.Conv2d(depth, depth, kernel_size=3, padding=1),
									nn.BatchNorm2d(depth),
									nn.ReLU(inplace=True))
		
		out_layer = nn.Sequential(nn.Conv2d(depth, 3, kernel_size=3, padding=1),
								   nn.Sigmoid())		

		self.layers = [in_layer]
		self.layers += [inter_layer]*depth
		self.layers += [out_layer]
		self.layers = nn.Sequential(*self.layers)

		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)

	def forward(self, x):
		y = x
		x = self.layers(x)
		return y - x