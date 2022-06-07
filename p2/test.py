
import torch,torchvision,utils,os
import numpy as np

from torchvision import datasets
from PIL import Image
from xml.dom.minidom import parse
from torchvision.transforms import functional as F

name_list = ['_background_','aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus','car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant', 'sheep', 'sofa','train', 'tvmonitor']
num_classes = len(name_list)


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
dataset_test = torch.utils.data.Subset(dataset_test, indices[-3000:])

data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,num_workers=1,collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(dataset_test,batch_size=1,shuffle=False,num_workers=1,collate_fn=utils.collate_fn)


print('\n Load the model...')
model = torch.load('pretrain_imagenet.pth')
model = model.to(device)

from engine import compute_map_class
compute_map_class(model,data_loader_test,device=device)
