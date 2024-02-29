import torch
from torch.utils import data
import os
from PIL import Image
from torchvision import transforms
import cv2
import random
# from utils import imageGenerate as IG
import torchvision

import numpy as np


class Dataset_segmentation(data.Dataset):
	def __init__(self, images_dir,mask_dir,image_size,random_crop=False,mask_list = ['urban_land', 'agriculture_land', 'rangeland', 'forest_land', 'water', 'barren_land', 'unknown']): 
		self.images_dir = images_dir
		self.mask_dir=mask_dir
		self.mask_list=mask_list
		self.image_size=image_size
		if type(image_size) !=type([]):
			self.image_size_h=image_size
			self.image_size_w=image_size
		else:
			self.image_size_h=image_size[1]
			self.image_size_w=image_size[0]

		self.list_IDs = os.listdir(self.images_dir)
		self.transform=transforms.Compose([transforms.ColorJitter(brightness=0.5), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])
		self.transform_mask=transforms.Compose([ transforms.ToTensor(),])
		
		self.random_crop=random_crop
		self.xmin=0
		self.xmax=0
		self.ymin=0
		self.ymax=0
		
	def __len__(self):
		return len(self.list_IDs)

	def generate_random_crop(self):
		self.xmin=random.randrange(0,1300)
		self.ymin=random.randrange(0,600)
		self.xmax=random.randrange(400,600)
		self.ymax=random.randrange(400,600)
		

	def read_image(self,image_path,thresholding=False):
		im = cv2.imread(image_path)
		try:
			h,w,_=im.shape
		except:
			print(image_path)
			h,w,_=im.shape
			
		if self.random_crop:
			im=im[self.ymin:min(self.ymin+self.ymax,h),self.xmin:min(self.xmin+self.xmax,w)]
		
		im = cv2.resize(im,(self.image_size_w,self.image_size_h))

		if thresholding:
			gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			im = cv2.threshold(gray, 45, 1, cv2.THRESH_BINARY)[1]
			# im=self.transform_mask(np.unsqueez(im))
			im=torch.tensor(im,).unsqueeze(0).float()

		else:
			im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
			im=self.transform(im)
		return im

	def read_image_and_mask(self,image_name,padding=False):
		if self.random_crop:
			self.generate_random_crop()
		img=self.read_image(os.path.join(self.images_dir,image_name))
		mask_list=[]

		for cat in self.mask_list:
			mask_path=os.path.join(self.mask_dir,cat,image_name[:-4]+'.png')
			mask=self.read_image(mask_path,thresholding=True)
			mask_list.append(mask)
		mask=torch.cat(mask_list,0)

		return img,mask
	def image_name(self,index):
		return self.list_IDs[index]

	def __getitem__(self, index):
		image_name = self.list_IDs[index]
		img,mask=self.read_image_and_mask(image_name)		
		return img,mask
