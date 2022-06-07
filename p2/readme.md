# Introduction
We train a Faster-RCNN on VOC2007 dataset, with Resnet-50 as its backbone. We consider three differnent approaches for training.

+ Initialize the backbone Resnet-50 with random parameters;
+ Use the pretrained Resnet-50 (on ImageNet);
+ Use the Resnet-50 of a pretrained Mask-RCNN (on COCO).

The module torchvision.models is used to download models. The commands below are corresponding to the three approaches, respectively.

```python
net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False) 
```

```python
net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True) 
```

```python
net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False) 
net_ = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
net.backbone = nn.Sequential(net_.backbone)
```

We mainly refer to https://javajgs.com/archives/45847. The model file can be downloaded from 

https://pan.baidu.com/s/1-OF-MkzxYYBQTXHPpctmdw?pwd=sjwl (pwd:sjwl).

# Usage
Run the command below to train your Faster-RCNN. Note that only one of 'random', 'imagenet' and 'coco' is selected.

```python
python train.py random/imagenet/coco
```

To test your trained Faster-RCNN

```python
python test.py
```python

The default number of test images is 3,000.


To do object detection on your own images, run


```python
python detect.py
```

remember to modify the path of .pth and .jpg files in the code. See our demos (by pretrain_coco.pth) in the folder user_detect.






