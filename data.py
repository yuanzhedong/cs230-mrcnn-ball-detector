import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
    def __getitem__(self, idx):
        '''
        Args:
            idx: index of sample to be fed
        return:
            dict containing:
            - PIL Image of shape (H, W)
            - target (dict) containing: 
                - boxes:    FloatTensor[N, 4], N being the n° of instances and it's bounding 
                boxe coordinates in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H;
                - labels:   Int64Tensor[N], class label (0 is background);
                - image_id: Int64Tensor[1], unique id for each image;
                - area:     Tensor[N], area of bbox;
                - iscrowd:  UInt8Tensor[N], True or False;
                - masks:    UInt8Tensor[N, H, W], segmantation maps;
        '''
        
        img_id = self.img_ids[idx]
        # List: get annotation id from coco
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        anns_obj = self.coco.loadAnns(ann_ids)
        # path for input image
        img_obj = self.coco.loadImgs(img_id)[0]
        path = img_obj['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))        
        bboxes = []
        for i in range(len(anns_obj)):
            xmin = anns_obj[i]['bbox'][0]
            ymin = anns_obj[i]['bbox'][1]
            xmax = xmin + anns_obj[i]['bbox'][2]
            ymax = ymin + anns_obj[i]['bbox'][3]
            bboxes.append([xmin, ymin, xmax, ymax])
        masks = [self.coco.annToMask(ann) for ann in anns_obj]
        areas = [ann['area'] for ann in anns_obj]

        boxes = torch.as_tensor(bboxes, dtype=torch.float32)

        labels = [obj["category_id"] for obj in anns_obj]
        labels = [self.json_category_id_to_contiguous_id[c] for c in labels]
        labels = torch.tensor(labels)

        #labels = torch.ones(len(anns_obj), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = torch.as_tensor(areas)
        iscrowd = torch.zeros(len(anns_obj), dtype=torch.int64)


        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


    def __len__(self):
        return len(self.img_ids)
