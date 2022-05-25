
We use *DeeplabV3 Plus* to do semantic segementation on a video from a driving recorder. We have referrence for https://github.com/VainF/DeepLabV3Plus-Pytorch.

We first transform the downloaded video to images by frames, which are inputs of the net. Then the output images are collected and formed as an intact video via PR.

The video after segementation (together with the original video and the model) can be downloaded from 
https://pan.baidu.com/s/1FU5zdLE70PIvWHe8QF2Gyw?pwd=sjwl (pwd:sjwl)

To test on your own video, first download the .pth model file to \checkpoints. Put the test images in \test_image, and type 

```python
python predict.py --input test_image  --dataset cityscapes --model deeplabv3plus_mobilenet --ckpt checkpoints/mydeeplab.pth --save_val_results_to test_results
```

The folder \demo shows some of the input and output images.
