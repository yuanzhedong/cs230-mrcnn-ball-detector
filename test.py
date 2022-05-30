import torch
from model import get_instance_segmentation_model
import utils
from torchvision.transforms import functional as F

from statistics import mode
import torch
import cv2
import random
import numpy as np
import transforms as T

from torch.optim.lr_scheduler import StepLR
from model import get_instance_segmentation_model
from data import CocoDataset
from engine import train_one_epoch, evaluate
import utils

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = len(utils.coco_names)

MODEL_PATH="./best_night_morning_aug/best.pth"

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
checkpointer = utils.Checkpointer(model)
checkpointer.load(MODEL_PATH)
#model.load_state_dict(torch.load(PATH))
model = checkpointer.model
model.eval()

model.to(device)

def infer_one_img(model, img):    
    masks, boxes, labels = utils.get_outputs(img, model, 0.965)
    orig_image = img.mul(255).permute(1, 2, 0).byte().numpy()
    result = utils.draw_segmentation_map(orig_image, masks, boxes, labels)
    return result

#cap = cv2.VideoCapture("./backyard/backyard.mp4")
cap = cv2.VideoCapture("./tenniscourt.mp4")
flag, frame = cap.read()
size = (int(cap.get(3)), int(cap.get(4)))

vw = cv2.VideoWriter('result_tc_aug.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
while flag:
    res = infer_one_img(model, F.to_tensor(frame))
    #cv2.imshow("res", res)
    vw.write(res)
    flag, frame = cap.read()