# Human-body-segmentation
## Overview
Image processing project for Human body segmentation, implemented in [PyTorch](https://github.com/pytorch/pytorch) and [JAX](https://github.com/google/jax).

## **Model used**

We have a pretrained model of FCN (Fully Convolutional Networks) with a Resnet101 backbone. 

Used  models urls:

"fcn_resnet101_coco": "https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth" ,

"resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth"

## **Content:**
### **Model surgery PyTorch**
We have made a model surgery in PyTorch through taking 4 layers of a pretrained RESNET101 and combining their output with FCN classifier to implement [create_feature_extractor](https://pytorch.org/vision/master/generated/torchvision.models.feature_extraction.create_feature_extractor.html#torchvision.models.feature_extraction.create_feature_extractor) manually. And we achieved the same result successfully.

### **Image Classification with FLAX-JAX**
Image classification through a pretrained JAX resnet100 model by using [FLAX](https://github.com/google/flax) Neural Network library.

### **Image Segmentation with FLAX-JAX**
We have made a model surgery using [FLAX](https://github.com/google/flax) ,through taking 4 layers from a pretrained RESNET101 as a backbone and combined them with a FCN classifier to create our model.
At the end, we applied segmentation mapping with our customized model to segment images.
