## Generator Architecture

- ```in_channels```:number of input channel.
- ```out_channels```:number of output channel.
- ```n_levels```: Depth of generator.
- ```padding```: Choice of padding the input,default set to False 
- ```batch_norm```: Choice of using batch normalization,default set to False
- ```up_mode```: Choice of choosing up sampling method. 
- ```n_channels```: same as```Latent_dim``` in ```config.json```file,Deepest layer output channels


The Generator architecture is a U-net variatent model [link](https://arxiv.org/pdf/1505.04597.pdf). Each layer of the U-net is composed of a double convolution, after each layer, the feature channel is doubled. After each layer, a residual connection is in place with the layer input. 

From the bottom convolution layer going up, we first up sample the input , three options are avaliable,```upconv```,```upsample```,```pixelshuffle```. Default option is ```upsample```. After upsampling the input, we do concatenation with the skip connection layers first before feeding into the up part of our network.

The last convolution is to transform U-net's output channels into our desired number of channels


## Discriminator Architecture

The Discriminator Architecture uses a similar structure in the first part of the generator, each layer is composed with a double convolution and the output channel is doubled. After 4 layers, two linear layers are connected, output shape is a 1 by 1 tensor representing the probability of the image is real or fake. 


## Loss function

- ```L1 loss```: The L1 difference between the network output and target image.
- ```BCE with logits Loss```: Default GAN loss.
- ```TV loss```: The total variation loss. 
- ```SSIM loss```: The structural similarity index measure.
- ```GMSD loss```: The Gradient Magnitude Similarity Deviation.


```L1 loss``` and  ```BCE with logits Loss``` are being used on default where the other three are optional loss can be turned on by setting the loss weight in  ```model_configs.json```.
