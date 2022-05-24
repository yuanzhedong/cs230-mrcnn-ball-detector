import torch
from model import get_instance_segmentation_model
import utils

num_classes = 3

PATH="./best_model/best.pth"
# get the model using our helper function


model = get_instance_segmentation_model(num_classes)
checkpointer = utils.Checkpointer(model)
checkpointer.load(PATH)
#model.load_state_dict(torch.load(PATH))
model = checkpointer.model
model.eval()