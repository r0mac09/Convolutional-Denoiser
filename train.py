import os

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data.dataloader import DataLoader

from src.model import Denoiser
from src.utils import ImDataset

if __name__ == "__main__":
	net = Denoiser(depth=20, conv_features=128).cuda()
	net.load_state_dict(torch.load('trained_model/denoiser.pth'))
	print(net)
	batch = 15
	dataset = ImDataset(input_folder='../Scaler/dataset/images1024x1024')
	loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=0)

	optimizer = optim.Adam(net.parameters(), lr=0.0005)

	criterion = nn.MSELoss()

	epochs = 10

	if not os.path.isdir('checkpoints'):
		os.makedirs('checkpoints')

	best_loss = 10000.
	net.train()
	for epoch in range(epochs):
		
		epoch_loss = 0.0
		for inputs, labels in loader:
			inputs = inputs.cuda()
			output = net(inputs)
			loss = criterion(output, labels.cuda())
			in_img = np.array(transforms.ToPILImage()(inputs.detach().cpu()[0]))
			out_img = np.array(transforms.ToPILImage()(output.detach().cpu()[0]))
			lb_img = np.array(transforms.ToPILImage()(labels[0]))
			cv.putText(out_img, f'loss: {loss.item()}', (20, 20), cv.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv.LINE_AA)
			cv.imshow('Input', cv.cvtColor(in_img, cv.COLOR_RGB2BGR))
			cv.imshow('Progress', cv.cvtColor(out_img, cv.COLOR_RGB2BGR))
			cv.imshow('Expected', cv.cvtColor(lb_img, cv.COLOR_RGB2BGR))
			batch_loss = loss.item()
			# print(batch_loss)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			epoch_loss += batch_loss/batch
			if cv.waitKey(1) ==  ord('q'):
				quit()
		epoch_loss /= len(loader)
		print(f'Epoch {epoch} done. Loss {epoch_loss}')

		torch.save(net.state_dict(), f'checkpoints/CP{epoch}.pth')

		if epoch_loss < best_loss:		
			torch.save(net.state_dict(), f'trained_model/denoiser.pth')
			best_loss = epoch_loss
