from tqdm import tqdm
from unet import UNet
import cv2
import torch
import glob
import numpy as np
import os
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
		self.transforms_img = transforms.Compose([transforms.ColorJitter(brightness=0.8),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])


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
			img = Image.fromarray(img)
			img = self.transforms_img(img)
			img = img.unsqueeze(0)
			img = img.cuda()
			
		masks_pred = self.net(img)
		return masks_pred.cpu().detach()
		

def write_imgs(mask_dir,img_name,image,h,w):
	for i, class_ in enumerate(['urban_land', 'agriculture_land', 'rangeland', 'forest_land', 'water', 'barren_land', 'unknown']):
		img=image[0,i,:,:]
		img=cv2.resize(img,(w,h))
		dir_path =os.path.join(mask_dir,class_)
		os.makedirs(dir_path, exist_ok=True)
		path=os.path.join(dir_path,img_name)
		cv2.imwrite(path,img)
		
	final_mask = np.zeros_like(image[0, 0, :, :])
	for i in range(0, image.shape[1]):
		final_mask +=image[0, i, :, :] 
	final_mask[final_mask>0] = 255 
 
	mask = np.logical_or(image[0, 0, :, :] >= 0.5, image[0, 1, :, :] >= 0.5)
	black = np.zeros((image.shape[-2], image.shape[-1], 3), dtype=np.uint8)
	black[mask] = np.array([255, 255, 255], dtype=np.uint8)
	
	dir_path =os.path.join(mask_dir,"result")
	os.makedirs(dir_path, exist_ok=True)
	path=os.path.join(dir_path,img_name)
	cv2.imwrite(path,black) 
		

if __name__ == '__main__':
	model_path="./TrainwtBeta02/best_CP.pth"
	ds=detect_segmentation(model_path, gpu=True)
	
	image_dir= "./data/test/imgs"
	mask_dir= "./data/test/pred_mask_dir"

	image_paths = glob.glob(image_dir+"/*.jpg")

	for image_path in tqdm(image_paths):
		img_name=image_path.split("/")[-1]
		
		img = cv2.imread(image_path)
		h, w = img.shape[:2]

		masks=ds.detect(img)
		mask=masks.numpy()
		
		mask[mask>=0.5]=255
		mask[mask<0.5]=0
		write_imgs(mask_dir,img_name,mask,h,w)
  