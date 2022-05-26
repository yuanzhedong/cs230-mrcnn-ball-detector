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

# utils



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# train_data_dir="./night_labels/train"
# train_coco="./night_labels/train/annotations.json"

# test_data_dir="./night_labels/test"
# test_coco="./night_labels/test/annotations.json"

train_data_dir="./night_morning_labels/train"
train_coco="./night_morning_labels/train/annotations.json"

test_data_dir="./night_morning_labels/test"
test_coco="./night_morning_labels/test/annotations.json"

# create own Dataset
train_dataset = CocoDataset(root=train_data_dir,
                          annotation=train_coco,
                          transforms=utils.get_transform(train=True)
                          )
test_dataset = CocoDataset(root=test_data_dir,
                          annotation=test_coco,
                          transforms=utils.get_transform(train=False)
                          )


# Batch size
train_batch_size = 1
test_batch_size = 1

# DataLoader
train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=train_batch_size,
                                          shuffle=True,
                                          num_workers=4,
                                          collate_fn=utils.collate_fn)
test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=test_batch_size,
                                          shuffle=True,
                                          num_workers=4,
                                          collate_fn=utils.collate_fn)

# our dataset has three classes only - background, ball and net
num_classes = 3

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

num_epochs = 50

train_iter = 0
eval_iter = 0
metric_logger = utils.TensorboardLogger(log_dir="./log", start_iter=0, delimiter=" ")
logdir = metric_logger.writer.logdir
checkpointer = utils.Checkpointer(model=model, optimizer=optimizer, scheduler=lr_scheduler, save_dir=logdir, save_to_disk=True)

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_iter = train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10, iter=train_iter, metric_logger=metric_logger)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    _, eval_iter = evaluate(model, test_data_loader, device=device, metric_logger = metric_logger, iter=eval_iter, checkpointer=checkpointer)



# img, _ = test_dataset[1]
# orig_image = img.mul(255).permute(1, 2, 0).byte().numpy()
# model.eval()
# masks, boxes, labels = get_outputs(img, model, 0.965)
# result = draw_segmentation_map(orig_image, masks, boxes, labels)
