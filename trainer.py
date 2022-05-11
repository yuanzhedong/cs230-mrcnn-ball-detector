from torch.optim.lr_scheduler import StepLR
import torch

from model import get_instance_segmentation_model
from data import CocoDataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_data_dir="./night_labels/train"
train_coco="./night_labels/train/annotations.json"

test_data_dir="./night_labels/test"
test_coco="./night_labels/test/annotations.json"
# create own Dataset
train_dataset = CocoDataset(root=train_data_dir,
                          annotation=train_coco,
                          transforms=get_transform(train=True)
                          )
test_dataset = CocoDataset(root=test_data_dir,
                          annotation=test_coco,
                          transforms=get_transform(train=False)
                          )

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

# Batch size
train_batch_size = 1
test_batch_size = 1

# own DataLoader
train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=train_batch_size,
                                          shuffle=True,
                                          num_workers=4,
                                          collate_fn=collate_fn)
test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=test_batch_size,
                                          shuffle=True,
                                          num_workers=4,
                                          collate_fn=collate_fn)





# our dataset has two classes only - background and person
num_classes = 2

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

num_epochs = 10

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, test_data_loader, device=device)
