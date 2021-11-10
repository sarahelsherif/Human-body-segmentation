# Human-body-segmentation
## Overview
Image processing project for Human body segmentation, implemented in Pytorch and [JAX](https://github.com/google/jax).

## **Model used**

We have a pretrained model of FCN Fully Convolutional Networks with a Resnet101 backbone. 

used  models urls:

"fcn_resnet101_coco": "https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth" ,

"resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth"

In addition to [PIX](https://github.com/deepmind/dm_pix/tree/a75741220b8c3ead32ff3e9d7d38eb315d5f0ed9) (image processing library in [JAX](https://github.com/google/jax))

## **Image Classification with FLAX-JAX**
Image classification through a pretrained JAX resnet100 model by using [FLAX](https://github.com/google/flax) Neural Network library.
