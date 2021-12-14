# Image Segmentation Project
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
## **Results**
**Input images:**

![bird](https://user-images.githubusercontent.com/83164531/146076681-266d3e67-0adf-40e5-bd1b-bfb3f447f148.png)

![cat](https://user-images.githubusercontent.com/83164531/146076863-a5ee3d63-07b7-4a55-898c-40d904afe5df.png)


**Segmented images:**

![segm bird JAX](https://user-images.githubusercontent.com/83164531/146075350-cdcf05e2-8241-4e3b-ba51-38ba79655270.png)

![segm cat](https://user-images.githubusercontent.com/83164531/146076539-03f23fde-6f35-417d-ab20-48e08e90e17c.png)

