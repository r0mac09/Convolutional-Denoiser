import os
import random

import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from cv2 import GaussianBlur, fastNlMeansDenoisingColored
from imgaug import augmenters as iaa
from PIL import Image
from torch.utils.data.dataset import Dataset

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO


class ImDataset(Dataset):
	def __init__(self, input_folder, sample=False):
		self.input_folder = input_folder if input_folder.endswith('/') else input_folder + '/'

		self.input_paths = [pth for pth in os.listdir(self.input_folder) if pth.endswith('.png')]

		self.image_transforms = [
			transforms.Resize((400, 400)),
			RandomNoise()
		]

		self.tensor_transforms = [
			transforms.ToTensor()
		]

		self.train_transforms = transforms.Compose(self.image_transforms + self.tensor_transforms)
		self.label_transforms = transforms.Compose([transforms.Resize((400, 400))] + self.tensor_transforms)

	def __getitem__(self, index):
		image = Image.open(self.input_folder + self.input_paths[index])
		t_image = self.train_transforms(image)
		l_image = self.label_transforms(image)
		return t_image, l_image

	def __len__(self): 
		return len(self.input_paths)

	def sample(self, num_samples=5):
		indices = np.random.randint(0, len(self), num_samples)
		sample_transforms = transforms.Compose(self.image_transforms)
		images = []

		for index in indices:
			images.append(Image.open(self.input_folder + self.input_paths[index]))

		samples = []
		for image in images:
			samples.append(sample_transforms(image))
		
		return samples

class RandomNoise(object):
	def __init__(self):
		self.gaussian = iaa.AdditiveGaussianNoise(loc=0, scale=0.04*255)
		self.poisson = iaa.AdditivePoissonNoise(lam=5.0, per_channel=True)
		if not os.path.isdir('tmp'):
			os.makedirs('tmp')

	def __call__(self, sample):
		im_arr = np.array(sample)

		im_arr = self.gaussian.augment_image(im_arr)
		im_arr = self.poisson.augment_image(im_arr)
		im_arr = GaussianBlur(im_arr, (3, 3), 0.0)
		im_arr = fastNlMeansDenoisingColored(im_arr, None, 6, 6, 4, 12)
		image = Image.fromarray(im_arr)

		return image
