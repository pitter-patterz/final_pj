import numpy as np
import torch, torchvision, cv2
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.transforms import functional as F

device = torch.device('cuda')
model = torch.load(r'./net/pretrain_imagenet.pth')
model.to(device)

print('\n Successfully load the model...')

name_list = ['_background_','aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

def showbbox(model,img,iou_thresh=0.3,prob_thresh=0.5):
       
    model.eval()
    with torch.no_grad():

        pred = model([img.to(device)])[0]
        keep = torchvision.ops.batched_nms(pred['boxes'],pred['scores'],pred['labels'],iou_threshold=iou_thresh)

        pred['boxes'] = pred['boxes'][keep]
        pred['labels'] = pred['labels'][keep]
        pred['scores'] = pred['scores'][keep]
                
    img = img.permute(1,2,0)  
    img = (img * 255).byte().data.cpu()  
    img = np.array(img)  
    
    for i in range(pred['boxes'].cpu().shape[0]):
        xmin = round(pred['boxes'][i][0].item())
        ymin = round(pred['boxes'][i][1].item())
        xmax = round(pred['boxes'][i][2].item())
        ymax = round(pred['boxes'][i][3].item())
        
        label = pred['labels'][i].item()
        prob = pred['scores'][i].item()
        
        if prob<prob_thresh:
          continue
        
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=2)
        cv2.putText(img, name_list[label]+', '+str(round(prob,2)), (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),thickness=1)

    save_path = 'detect'+str(np.random.randint(int(1e10)))+'.jpg'
    plt.figure(figsize=(20,15))
    plt.imshow(img)
    plt.savefig(save_path)

    print('\nThe image is saved at',save_path)


img = Image.open(r'./user_detect/demo2.jpg').convert("RGB")
img = F.pil_to_tensor(img)/255.0
showbbox(model,img)







