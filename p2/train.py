import torch,torchvision,utils
import os,sys
import numpy as np

from tqdm import tqdm
from torchvision import datasets
from PIL import Image
from xml.dom.minidom import parse
from torchvision.transforms import functional as F

name_list = ['_background_','aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus','car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant', 'sheep', 'sofa','train', 'tvmonitor']
num_classes = len(name_list)


method = str(sys.argv[-1])
if method not in ['random','imagenet','coco']:
    print('\nPlease check the methods...')
    sys.exit()


class MarkDataset(torch.utils.data.Dataset):
    def __init__(self,root):
        self.root = root
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.bbox_xml = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
 
    def __getitem__(self, idx):
        # load images and bbox
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        bbox_xml_path = os.path.join(self.root, "Annotations", self.bbox_xml[idx])
        img = Image.open(img_path).convert("RGB")      
        

        dom = parse(bbox_xml_path)

        data = dom.documentElement

        objects = data.getElementsByTagName('object')        

        boxes = []
        labels = []
        for object_ in objects:

            name = object_.getElementsByTagName('name')[0].childNodes[0].nodeValue 
            labels.append(np.int(name_list.index(name)))  
            bndbox = object_.getElementsByTagName('bndbox')[0]
            xmin = np.float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
            ymin = np.float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
            xmax = np.float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
            ymax = np.float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
            boxes.append([xmin, ymin, xmax, ymax])        
 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)        
 
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(objects),), dtype=torch.int64)
 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        img = F.pil_to_tensor(img)/255.0

        return img, target
 
    def __len__(self):
        return len(self.imgs)



root = r'./VOC2007/'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print('\n Prepare datasets...')

dataset = MarkDataset(root)
dataset_test = MarkDataset(root)
indices = torch.randperm(len(dataset)).tolist()

dataset = torch.utils.data.Subset(dataset, indices[:4000])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-1000:])

data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,num_workers=1,collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(dataset_test,batch_size=1,shuffle=False,num_workers=1,collate_fn=utils.collate_fn)


print('\n Load the model...')
print('\n The method being applied:',method)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=False)

if method == 'imagenet':
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)

if method == 'coco':
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=False)
    mask = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True)
    model.backbone = torch.nn.Sequential(mask.backbone)

model = model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params,lr=1e-5,weight_decay=5e-4)

from engine import train_one_epoch, compute_map

total_epochs = 12
for epoch in range(total_epochs):

    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=20)        
    model.train()
    with torch.no_grad():
        test_loss,test_num = 0.0,0
        for k,(images,targets) in tqdm(enumerate(data_loader_test)):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            out = model(images,targets)
            test_loss += sum(out.values()).item()
            test_num += 1
        print('\n\nLoss on test images',test_loss/test_num)  

    compute_map(model,data_loader_test,device=device)

    for param in optimizer.param_groups:
        param['lr'] *= 0.8
    print('')
    print('==================================================')
    print('')

print("That's it!")

# torch.save(model,'net.pth')
