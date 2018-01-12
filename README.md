# Progressive GAN in PyTorch
Implementation of Progressive Growing of GANs (https://arxiv.org/abs/1710.10196) in PyTorch

Currently implemented and tested up to 128x128 images.

Usage:

> python train.py -d {celeba, lsun} PATH
  
Currently CelebA and LSUN dataset is supported. (Warning: Using LSUN dataset requires vast amount of time for creating index cache.)

## Sample

* Sample from the model trained on CelebA 

![Sample of the model trained on CelebA](doc/celeba_570600.png)

* Sample from the model trained on LSUN (dog)

![Sample of the model trained using LSUN (dog)](doc/lsun_600000.png)
