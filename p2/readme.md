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
To train the Faster-RCNN

```python
python train.py random/imagenet/coco
```

