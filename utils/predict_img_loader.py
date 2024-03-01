from torch.utils import data
from torchvision import transforms
import cv2

class predict_img_generator(data.Dataset):
	def __init__(self,image_size,batch_size=4):
		self.image_size=image_size
		self.batch_size=batch_size
		self.image_size_crop=1500
		self.transform=transforms.Compose([transforms.ToTensor(),  transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])
		self.transform_mask=transforms.Compose([ transforms.ToTensor()])
		

	def generate_img_stack(self,img_path):
		X= cv2.imread(img_path)
		X=cv2.cvtColor(X,cv2.COLOR_BGR2RGB)

		h,w,_=X.shape
		nh=500
		nw=500
		X=cv2.resize(X,(nw,nh))
		X= self.transform(X)
		X=X.unsqueeze(0)

		return X,h,w,nh,nw
