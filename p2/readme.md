We train three different Faster-RCNNs on VOC dataset.

+ Initialize the backbone net (VGG16) with random parameters.

+ Use the VGG16 pretrained on ImageNet.

+ Use the mask-RCNN trained on coco.

The implementation of 1,2 is based on our *project 2-2*, see https://github.com/pitter-patterz/pj2-2.

We have referrence for

