import math
import sys
import time
import torch
import cv2
import torchvision.models.detection.mask_rcnn
import random

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, iter, metric_logger):
    model.train()
    #metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(iter, loss=losses_reduced, **loss_dict_reduced, lr=optimizer.param_groups[0]["lr"])

        iter += 1
    return iter


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

import numpy as np
coco_names=["_ignore", "ball", "net"]
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

def get_outputs(outputs, threshold):
    res=[]
    for output in outputs:
        # get all the scores
        scores = list(output['scores'].detach().cpu().numpy())
        # index of those scores which are above a certain threshold
        thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
        thresholded_preds_count = len(thresholded_preds_inidices)
        # get the masks
        masks = (output['masks']>0.5).squeeze().detach().cpu().numpy()
        # discard masks for objects which are below threshold
        masks = masks[:thresholded_preds_count]
        # get the bounding boxes, in (x1, y1), (x2, y2) format
        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in output['boxes'].detach().cpu()]
        # discard bounding boxes below threshold value
        boxes = boxes[:thresholded_preds_count]
        # get the classes labels
        labels = [coco_names[i] for i in output['labels']]
        res.append((masks, boxes, labels))
    return res

def draw_segmentation_map(image, masks, boxes, labels):
    alpha = 1 
    beta = 0.6 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)
        # apply a randon color mask to each object
        color = COLORS[random.randrange(0, len(COLORS))]
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = color
        # combine all the masks into a single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        #convert the original PIL image into NumPy format
        image = np.array(image)
        # convert from RGN to OpenCV BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # apply mask on the image
        cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
        # draw the bounding boxes around the objects
        cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color, 
                      thickness=2)
        # put the label text above the objects
        cv2.putText(image , labels[i], (boxes[i][0][0], boxes[i][0][1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 
                    thickness=2, lineType=cv2.LINE_AA)
    
    return image
@torch.no_grad()
def evaluate(model, data_loader, device, metric_logger, iter, checkpointer):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    #metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    viz_imgs=[]

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

        pred_labels = get_outputs(outputs, 0.965)
        for i in range(len(images)):
            orig_img = images[i].mul(255).permute(1, 2, 0).byte().cpu().numpy()
            try:
                viz_img = draw_segmentation_map(orig_img, *pred_labels[i])
            except:
                viz_img = np.array(orig_img)
                viz_img = cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR)
                
                cv2.putText(viz_img , "FAILED", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 
                    thickness=2, lineType=cv2.LINE_AA)
                h,w, _ = viz_img.shape
                cv2.rectangle(viz_img, (0, 0), (w-1, h-1), (255, 0, 0), 2)
            viz_imgs.append(torch.from_numpy(viz_img).permute(2,0,1))

        metric_logger.update(iter, model_time=model_time, evaluator_time=evaluator_time)
        iter += 1
    grid = torchvision.utils.make_grid(viz_imgs)
    metric_logger.add_image(grid, iter)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    import pdb
    pdb.set_trace()
    box_mAP = coco_evaluator.stats[-2] #
    segm_mAP = coco_evaluator.stats[-1]
    curr_avg_loss = metric_logger.loss.global_avg
    #checkpointer.save(f"best_loss_{curr_avg_loss}", loss=curr_avg_loss)
    checkpointer.save(f"best", loss=curr_avg_loss, box_mAP = box_mAP, segm_mAP = segm_mAP)
    torch.set_num_threads(n_threads)
    return coco_evaluator, iter
