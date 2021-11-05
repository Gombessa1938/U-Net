## Generator Architecture

- ```in_channels```:number of input channels.
- ```out_channels```:number of output channels.
- ```n_levels```: Depth of generator.
- ```padding```: Padding after Conv layers, default set to False 
- ```batch_norm```: Batch normalization, default set to False
- ```up_mode```: Upsampling method in the U-net Up blocks. [upconv, upsample(default), pixelshuffle]
- ```n_channels```: ```Latent_dim``` in ```config.json```file Number of embedding channels in the deepest layer.


The Generator architecture is a U-net variatent model modified from [link](https://arxiv.org/pdf/1505.04597.pdf).

Each level of the U-net downsampling path is composed of convolutional blocks with residual connections. Each block is composed of two convolutional layers. After each block, the feature channel is doubled. After each block, a residual connection is in place with the block input. 

Each block in the upsampling path is similiar to that in the downsampling path. From the bottom convolution layer going up, the input feature map is first upsampled Three options are avaliable:```upconv```,```upsample```,```pixelshuffle```. The default option is ```upsample```. After upsampling the input, it is concatenated with the skip connection layer output from the sysmetric downsampling block and fed into the corresponding upsampling block.

The last convolution is to transform U-net's output channels into desired number of channels and match the label images.


## Discriminator Architecture

The Discriminator Architecture uses a similar structure in the first part of the generator, each layer is composed with a double convolution and the output channel is doubled. After 4 layers, two linear layers are connected, output shape is a 1 by 1 tensor representing the probability of the image is real or fake. 
