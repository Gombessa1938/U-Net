import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F


'''
Res-U net with soft attension gate

'''


class DoubleConv(nn.Module):                                             
	def __init__(self,in_channels,out_channels,bottom=False):
		super(DoubleConv,self).__init__()
		if bottom == False:
			middle_channels = (in_channels + out_channels)//2
		else:
			middle_channels = in_channels
		
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels,middle_channels ,3,1,1,bias=True),
			#nn.BatchNorm2d(middle_channels),
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			nn.Dropout2d(p=0.0, inplace=False),
			
			nn.Conv2d(middle_channels,out_channels,3,1,1,bias=True),
			#nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(negative_slope=0.1, inplace=True),)

	def forward(self,x):
		out = self.conv(x)

		diff = out.shape[1] - x.shape[1]
		temp = F.pad(x, (0, 0, 0, 0, 0, diff))  
		
		return out + temp #residual block



class Attension_gate(nn.Module):
    def __init__(self,F_g,F_x,F_int):
        super(Attension_gate,self).__init__()

        self.W_g = nn.Sequential(nn.Conv2d(F_g,F_int,kernel_size=1,stride=1,padding=0,bias=True),
                                 nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_x,F_int,kernel_size=1,stride=2,padding=0,bias=True),
                                 nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int,1,kernel_size=1,stride=1,padding=0,bias=True),
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())
        self.resample = nn.Upsample(mode='bilinear',
                                                scale_factor=2,
                                                align_corners=False)
        self.relu = nn.ReLU(inplace=True)


    def forward(self,g,x):

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        assert g1.shape == x1.shape,'check g,x,shape'
       
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        out = self.resample(psi)
        out = out*x 
        return out 



class UNet_Generator(nn.Module):
	def __init__(self,in_channels = 3,out_channels = 3,features=[32,64,128,256,512]):
		super(UNet_Generator,self).__init__()
		
		self.first = nn.Sequential(nn.Conv2d(in_channels,16,3,1,1),
								   nn.LeakyReLU(0.1,inplace=True))
		self.ups=nn.ModuleList()
		self.downs = nn.ModuleList()

		self.DoubleConv_in = 16
		#going down
		for feature in features:
			self.downs.append(DoubleConv(self.DoubleConv_in,feature))
			self.DoubleConv_in = feature


		#bottom layer
		self.bottom = DoubleConv(features[-1],features[-1],bottom=True) 

		#going up
		self.upsample = nn.Sequential(nn.Upsample(mode='bilinear',
                                                scale_factor=2,
                                                align_corners=False))
		for feature in reversed(features):
			self.ups.append(DoubleConv(feature*2,feature//2,bottom=False))

		
		self.final_conv = nn.Conv2d(features[0]//2,out_channels,kernel_size=1)
		self.tanh = nn.Tanh()


	def forward(self,x):
		
		skip_connections = []
		x = self.first(x)
		for layers in self.downs:
			x = layers(x)

			skip_connections.append(x)
			x = F.avg_pool2d(x, 2) #average pool stride 2
		

		x = self.bottom(x)

		skip_connections = skip_connections[::-1] #reverse the list

		g = x #output from bottom layer we use as g

		for idx in range(0,len(self.ups)):
			#upsample first, concat, then upconv
			x = self.upsample(x)
		
			skip_connection = skip_connections[idx]

			if x.shape != skip_connection.shape:
				x = TF.resize(x,size=skip_connection.shape[2:])  #resize ,[2:] get the current shape

			self.attension_gate = Attension_gate(g.shape[1],x.shape[1],x.shape[1])
			skip_with_attension = self.attension_gate(g,skip_connection)


			concat_skip  = torch.cat((skip_connection,x),dim=1)
			x = self.ups[idx](concat_skip)
			
			g = x  #resign g as previous layer output	

		x = self.final_conv(x)
		x = self.tanh(x)

		return x




class Discriminator(nn.Module):
	def __init__(self,in_channels,conv_inchannels,features=[32,64,128,256,512]):
		super(Discriminator,self).__init__()

		self.first = nn.Sequential(nn.Conv2d(in_channels,16,kernel_size=3,stride=1,padding = 1),
								   nn.LeakyReLU(0.1,inplace=True))

		self.downs = nn.ModuleList()
		
		for feature in features:
			self.downs.append(DoubleConv(conv_inchannels,feature,bottom=False))
			conv_inchannels = feature

		self.center = nn.Sequential(nn.Conv2d(512,512,kernel_size=3,stride=2,padding=1),
									nn.LeakyReLU(0.1,inplace=True),
									nn.Dropout2d(p=0.0,inplace=False),
									nn.Conv2d(512,512,kernel_size=3,stride=2,padding=1),
									nn.LeakyReLU(0.1,inplace=True))

		self.last_layer = nn.Sequential(nn.Linear(2048,1024),
										nn.LeakyReLU(0.1,inplace=True),
										nn.Dropout(p=0.5),
										nn.Linear(1024,1))

	def forward(self,x):
		
		x = self.first(x)
		
		for layers in self.downs:
			
			x = layers(x)
			x = F.avg_pool2d(x, 2)
			
		x = self.center(x)  #(batchsize,512,8,8)-->(batchsize,512,2,2)

		x = x.view(x.size(0), -1)

		x = self.last_layer(x)

		return x


if __name__=='__main__':

    model = Attension_gate(F_g=64,F_x=128,F_int=128)
    g = torch.rand((1,64,64,64))
    x = torch.rand((1,128,128,128))
    out = model(g,x)
    test = torch.rand((1,2,3,4))


    x = torch.rand((5,3,256,256))
    gen = UNet_Generator(3,3)
    out = gen(x)

    dis = Discriminator(3,16)
    x = torch.rand((5,3,256,256))
    out = dis(x)
    print(out.shape)