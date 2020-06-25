import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data.dataloader import DataLoader

from src.model import Denoiser
from src.utils import ImDataset

if __name__ == "__main__":
	net = Denoiser(depth=12).cuda()
	print(net)
	batch = 25
	dataset = ImDataset(input_folder='../Scaler/dataset/images1024x1024')
	loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=0)

	print(list(net.parameters()))

	optimizer = optim.SGD(net.parameters(),
						  lr=0.05,
						  momentum=0.9,
						  weight_decay=0.0005)

	criterion = nn.MSELoss()

	epochs = 10

	if not os.path.isdir('checkpoints'):
		os.makedirs('checkpoints')

	best_loss = 10000.

	for epoch in range(epochs):
		net.train()
		epoch_loss = 0.0
		for inputs, labels in loader:
			inputs = inputs.cuda()
			output = net(inputs)
			loss = criterion(output, labels.cuda())
			batch_loss = loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			epoch_loss += batch_loss
		epoch_loss /= len(loader)
		print(f'Epoch {epoch} done. Loss {epoch_loss}')

		if epoch_loss < best_loss:
			torch.save(net.state_dict(), f'checkpoints/CP{epoch}.pth')
			torch.save(net.state_dict(), f'trained_model/denoiser.pth')
			best_loss = epoch_loss