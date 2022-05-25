We train three different Faster-RCNNs on VOC dataset.

1> Initialize the backbone net (VGG16) with random parameters.

2> Use the VGG16 pretrained on ImageNet (from torchvision.models).

3> Use the mask-RCNN trained on coco.

The implementation of 1,2 is based on our *project 2-2*, see https://github.com/pitter-patterz/pj2-2.

We have referrence for https://github.com/chenyuntc/simple-faster-rcnn-pytorch.

