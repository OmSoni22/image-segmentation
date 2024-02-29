from tqdm import tqdm
from unet import UNet
import cv2
import torch
import glob
import numpy as np
import os
from utils import predict_img_loader
import random
from torchvision import transforms
from PIL import Image

class detect_segmentation(object):
	"""docstring for detect_segmentation"""
	def __init__(self, model_path,threshold=0.5,gpu=True):
		super(detect_segmentation, self).__init__()
		self.model_path = model_path
		self.threshold=threshold
		self.gpu=gpu

		self.net = UNet(n_channels=3, n_classes=7)
		self.load_model_para(self.model_path)		

		if self.gpu:
			self.net=self.net.cuda()

		self.net=self.net.eval()

	def load_model_para(self,path):
		self.net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
		print('Model loaded from {}'.format(path))

	def detect(self,img):
		if self.gpu:
			img = img.cuda()
			
		masks_pred = self.net(img).cpu().detach()
		return masks_pred
		

def write_imgs(mask_dir,img_name,image,h,w):
	for i, class_ in enumerate(['urban_land', 'agriculture_land', 'rangeland', 'forest_land', 'water', 'barren_land', 'unknown']):
		img=image[0,i,:,:]
		img=cv2.resize(img,(w,h))
		dir_path =os.path.join(mask_dir,class_)
		os.makedirs(dir_path, exist_ok=True)
		path=os.path.join(dir_path,img_name)
		cv2.imwrite(path,img)


if __name__ == '__main__':
	model_path="./TrainwtBeta02/best_CP.pth"

	ds=detect_segmentation(model_path, gpu=True)
	pg=predict_img_loader.predict_img_generator(500,batch_size=32)

	image_dir= "./data/test/imgs"
	mask_dir= "./data/test/pred_mask_dir"

	image_paths = glob.glob(image_dir+"/*.jpg")

	for image_path in tqdm(image_paths):
		img_name=image_path.split("/")[-1]
		
		img_stack,h,w,nh,nw=pg.generate_img_stack(image_path)
		
		masks=ds.detect(img_stack)
		mask=masks.numpy()
		
		mask[mask>=0.5]=255
		mask[mask<0.5]=0
		write_imgs(mask_dir,img_name,mask,h,w)
  