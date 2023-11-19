###########################################################
# This script tests a modified lenet5.
###########################################################

import numpy as np # linear algebra
import os
import glob
from matplotlib import cm
from matplotlib import pyplot as plt
import cv2
import util
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import utils,models
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import trange
from time import sleep
from scipy.io import loadmat
import torchvision.datasets as dset
from torch.utils.data import sampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
#from partialconv2d import PartialConv2d
from model import self2self
import matplotlib.pyplot as plt 
import random


class soft(nn.Module):
	def __init__(self):
		super(soft, self).__init__()

	def forward(self, x, lam):
		x_abs = x.abs() - lam
		zeros = x_abs - x_abs
		n_sub = torch.max(x_abs, zeros)
		x_out = torch.mul(torch.sign(x), n_sub)
		return x_out


soft_thres = soft()


def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()

def getpsnr(img1,img2):
	mse = np.mean(np.square(img1-img2))
	imgmax = np.square(abs(img1).max())
	psnr = 10*np.log10(imgmax/mse)
	return psnr



def image_loader(image, device):
	image = image.transpose(3, 0, 1, 2)
	image = image.astype(np.float32)
	image = torch.tensor(image)
	image = image.float()
	image = image.unsqueeze(1)
	return image.to(device)


if __name__ == "__main__":
	##Enable GPU
	USE_GPU = True
	
	dtype = torch.float32
	
	if USE_GPU and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	
	print('using device:', device)
	model = self2self(1,0.3)
	img = np.load('data.npy')
	imgmax = img.max()
	imgmin = img.min()
	img = (img-imgmin)/(imgmax-imgmin)
	img = img[:,:,:,np.newaxis]
	learning_rate = 1e-4
	torch.cuda.set_device(0)
	model = model.cuda()
	

	optimizer = optim.Adam(model.parameters(), lr = learning_rate)
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 2000, gamma=0.8)
	t,w,h,c = img.shape
	p=0.4
	NPred=100
	thres = 0.1
	mu = 0.00001

	for itr in range(5000):
		p_mtx = np.random.uniform(size=[1, w, h, c])
		p_mtx = np.repeat(p_mtx, t, axis=0)
		mask = (p_mtx>p).astype(np.double)
		img_input = img
		y = img
		img_input_tensor = image_loader(img_input, device)
		
		y = image_loader(y, device)
		
		mask = np.expand_dims(np.transpose(mask,[3,0,1,2]),0)
		mask = torch.tensor(mask).to(device, dtype=torch.float32)
		

		model.train()
		img_input_tensor = img_input_tensor*mask
		img_input_tensor = model(img_input_tensor, mask)
		loss = torch.norm((img_input_tensor-y)*(1-mask),2)



		# non-local patch
		x = torch.squeeze(img_input_tensor).detach().cpu().numpy()
		T, H, W = x.shape
		CSize = 6
		patchsize = 12
		mse = []
		rnd_t = np.random.randint(0, T - CSize + 1)
		rnd_h = np.random.randint(0, H - CSize + 1)
		rnd_w = np.random.randint(0, W - CSize + 1)
		center = x[rnd_t: rnd_t + CSize, rnd_h: rnd_h + CSize, rnd_w: rnd_w + CSize]
		for k in range(max(rnd_t - patchsize, 0), min(rnd_t + patchsize, T - CSize)+1):
			for i in range(max(rnd_h - patchsize, 0), min(rnd_h + patchsize, H - CSize)+1):
				for j in range(max(rnd_w - patchsize, 0), min(rnd_w + patchsize, W - CSize)+1):
					other = x[k: k + CSize, i: i + CSize, j: j + CSize]
					mse.append(np.sum(
						np.square(center - other)))
		parnum = 70
		amse = mse.copy()
		amse.sort()
		amse = amse[:parnum]
		index = []
		for i in amse:
			index.append(mse.index(i))
		rankpatch = torch.zeros((parnum, CSize, CSize, CSize), device=img_input_tensor.device)
		for i in range(parnum):
			rankpatch[i, :, :, :] = img_input_tensor[0, 0,      max(rnd_t - patchsize, 0) + index[i] // ((min(rnd_h + patchsize, H - CSize) - max(rnd_h - patchsize, 0) + 1)*(min(rnd_w + patchsize, W - CSize) - max(rnd_w - patchsize, 0) + 1))
																:max(rnd_t - patchsize, 0) + index[i] // ((min(rnd_h + patchsize, H - CSize) - max(rnd_h - patchsize, 0) + 1)*(min(rnd_w + patchsize, W - CSize) - max(rnd_w - patchsize, 0) + 1)) + CSize
			,      max(rnd_h - patchsize, 0) + index[i] % ((min(rnd_h + patchsize, H - CSize) - max(rnd_h - patchsize, 0) + 1)*(min(rnd_w + patchsize, W - CSize) - max(rnd_w - patchsize, 0) + 1)) // (min(rnd_w + patchsize, W - CSize) - max(rnd_w - patchsize, 0) + 1)
				   :max(rnd_h - patchsize, 0) + index[i] % ((min(rnd_h + patchsize, H - CSize) - max(rnd_h - patchsize, 0) + 1)*(min(rnd_w + patchsize, W - CSize) - max(rnd_w - patchsize, 0) + 1)) // (min(rnd_w + patchsize, W - CSize) - max(rnd_w - patchsize, 0) + 1) + CSize
			,      max(rnd_w - patchsize, 0) + index[i] % ((min(rnd_h + patchsize, H - CSize) - max(rnd_h - patchsize, 0) + 1)*(min(rnd_w + patchsize, W - CSize) - max(rnd_w - patchsize, 0) + 1)) % (min(rnd_w + patchsize, W - CSize) - max(rnd_w - patchsize, 0) + 1)
				   :max(rnd_w - patchsize, 0) + index[i] % ((min(rnd_h + patchsize, H - CSize) - max(rnd_h - patchsize, 0) + 1)*(min(rnd_w + patchsize, W - CSize) - max(rnd_w - patchsize, 0) + 1)) % (min(rnd_w + patchsize, W - CSize) - max(rnd_w - patchsize, 0) + 1) + CSize]
		rankpatch = rankpatch.view(parnum, CSize ** 3)


		#NN
		#u, s, d = torch.svd(rankpatch)

		#TV
		#s = rankpatch[1:, :] - rankpatch[:-1, :]

		#CTV
		D_h_ = rankpatch[1:, :] - rankpatch[:-1, :]
		u, s, d = torch.svd(D_h_)

		D_h = s.clone().detach()
		if itr == 0:
			global D_1, V_1

			D_1 = torch.zeros(s.size()).cuda()

			thres_tv = thres * torch.ones(s.size()).cuda()

			V_1 = D_h.type(dtype)

		V_1 = soft_thres(D_h + D_1 / mu, thres_tv)
		#loss = torch.norm((img_input_tensor - y) * (1 - mask), 2)
		loss = loss + mu / 2 * torch.norm(s - (V_1 - D_1 / mu), 2)




		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		lr_scheduler.step()
		D_1 = (D_1 + mu * (D_h  - V_1)).clone().detach()



		print("iteration %d, loss = %.4f" % (itr+1, loss.item()*100))

		if (itr+1)%2500 == 0:
			model.eval()
			sum_preds = np.zeros((img.shape[0],img.shape[1],img.shape[2],img.shape[3]))
			for j in range(NPred):
				p_mtx = np.random.uniform(size=[1, w, h, c])
				p_mtx = np.repeat(p_mtx, t, axis=0)
				mask = (p_mtx>p).astype(np.double)
				img_input = img*mask
				img_input_tensor = image_loader(img_input, device)
				
				mask = np.expand_dims(np.transpose(mask,[3,0,1,2]),0)
				mask = torch.tensor(mask).to(device, dtype=torch.float32)
				
				img_input_tensor = model(img_input_tensor,mask)
				sum_preds[:,:,:,:] += np.transpose(img_input_tensor.detach().cpu().numpy(),[2,3,4,1,0])[:,:,:,:,0]
			avg_preds = np.squeeze(sum_preds/NPred)
			write_img = avg_preds
			write_img = np.array(write_img)
			write_img = write_img*(imgmax-imgmin)+imgmin
			print("iteration %d, loss = %.4f" % (itr+1, loss.item()*100))
			np.save('denoised'+str(itr+1)+'.npy', write_img)
			#apsnr = getpsnr(original,write_img)
			#print(apsnr)
