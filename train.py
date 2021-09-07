import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNet
from utils import(
	load_checkpoint,
	save_checkpoint,
	get_loaders,
	save_predictions_as_imgs,
	check_accuracy,
	)

#hyperparameters:


learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else'cpu'
batch_size = 32
num_epochs = 1
num_workers = 2
image_height = 160  #1280 og
image_width = 240 #1918 og
pin_memory = True 
load_model = False
train_img_dir = 'train_imges/'
train_mask_dir = 'train_masks/'
val_img_dir = 'val_images/'
val_mask_dir='val_masks/'

def train_fn(loader,model,optimizer,loss_fn,scaler):
	loop = tqdm(loader)

	for batch_idx,(data,targets) in enumerate(loop):
		data = data.to(device)
		targets = targets.float().unsqueeze(1).to(device=device)

		with torch.cuda.amp.autocast():
			predictions = model(data)
			loss = loss_fn(predictions,targets)

		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()



		#tqdmloop
		loop.set_postfix(loss=loss.item())





def main():
	train_transform = A.Compose(

		[ 

		A.Resize(height=image_height,width = image_width),
		A.Rotate(limit=35,p=1.0),
		A.HorizontalFlip(p=0.5),
		A.VerticalFlip(p=0.1),
		A.Normalize(
			mean=[0.0,0.0,0.0],
			std=[1.0,1.0,1.0],
			max_pixel_value = 255.0
			),
		ToTensorV2(), ] )

	val_transforms = A.Compose(

		[ 

		A.Resize(height=image_height,width = image_width),
		A.Normalize(
			mean=[0.0,0.0,0.0],
			std=[1.0,1.0,1.0],
			max_pixel_value = 255.0
			),
		ToTensorV2(), ] )

	model = UNet(in_channels=3,out_channels=1).to(device)

	loss_fn = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(),lr=learning_rate)


	train_loader,val_loader = get_loaders(
		train_img_dir,
		train_mask_dir,
		val_img_dir,
		val_mask_dir,
		batch_size,
		train_transform,
        val_transforms,
		num_workers,
		pin_memory,
	)
	scaler = torch.cuda.amp.GradScaler()
	for epoch in range(num_workers):
		train_fn(train_loader,model,optimizer,loss_fn,scaler)

		#save
		#check accuracy
		checkpoint = {
			'state_dic':model.state_dict(),
			'optimizer':optimizer.state_dict(),
		}
		save_checkpoint(checkpoint)

		check_accuracy(val_loader,model,device=device)

		save_predictions_as_imgs(
			val_loader,model,folder='saved_images/',device=device)



if __name__=='__main__':
	main()

