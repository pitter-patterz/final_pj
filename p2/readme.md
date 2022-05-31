# Introduction
We train a Faster-RCNN on VOC2007 dataset, with Resnet-50 as its backbone. We consider three differnent approaches for training.

+ Initialize the backbone Resnet-50 with random parameters;
+ Use the pretrained Resnet-50 (on ImageNet);
+ Use the Resnet-50 of a pretrained Mask-RCNN (on COCO).

