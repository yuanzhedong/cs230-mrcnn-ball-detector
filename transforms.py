import random
import torch

from torchvision.transforms import functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image.copy())
        return image, target


# import numpy as np
# import imgaug
# def extract_bboxes(mask):
#     """Compute bounding boxes from masks.
#     mask: [num_instances, height, width]. Mask pixels are either 1 or 0.
#     Returns: bbox array [num_instances, [x1, y1, x2 y2]].
#     """
#     boxes = np.zeros([mask.shape[0], 4], dtype=np.int32)
#     for i in range(mask.shape[0]):
#         m = mask[i, :, :]
#         # Bounding box.
#         vertical_indicies = np.where(np.any(m, axis=0))[0]
#         horizontal_indicies = np.where(np.any(m, axis=1))[0]
#         if horizontal_indicies.shape[0]:
#             x1, x2 = horizontal_indicies[[0, -1]]
#             y1, y2 = vertical_indicies[[0, -1]]
#             # x2 and y2 should not be part of the box. Increment by 1.
#             x2 += 1
#             y2 += 1
#         else:
#             # No mask for this instance. Might happen due to
#             # resizing or cropping. Set bbox to zeros
#             x1, x2, y1, y2 = 0, 0, 0, 0
#             print("###############")
#         boxes[i] = np.array([x1, y1, x2, y2])
#     return boxes.astype(np.int32)

import copy
# class AugTransforms(object):
#     def __init__(self, augmentation):
#         self.augmentation = augmentation

#     def __call__(self, image, target):
        
#         orig_image = copy.deepcopy(image)
#         if "boxes" not in target.keys():
#             return image, target
#         mask = target["masks"]
#         bbox = target["boxes"]
#         # Augmentation
#         # This requires the imgaug lib (https://github.com/aleju/imgaug)
#         if self.augmentation:
#             import imgaug

#             # Augmenters that are safe to apply to masks
#             # Some, such as Affine, have settings that make them unsafe, so always
#             # test your augmentation on masks
#             MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
#                                "Fliplr", "Flipud", "CropAndPad",
#                                "Affine", "PiecewiseAffine"]

#             def hook(images, augmenter, parents, default):
#                 """Determines which augmenters to apply to masks."""
#                 return augmenter.__class__.__name__ in MASK_AUGMENTERS

#             # Store shapes before augmentation to compare
#             image = np.asarray(image)
#             mask = mask.to('cpu').numpy()
#             mask=np.transpose(mask, (1, 2, 0))
#             image_shape = image.shape
#             mask_shape = mask.shape
            
#             # Make augmenters deterministic to apply similarly to images and masks
#             det = self.augmentation.to_deterministic()
#             image = det.augment_image(image)
#             # Change mask to np.uint8 because imgaug doesn't support np.bool
# #             for i in range(mask.shape[0]):
# #                 mask[i] = det.augment_image(mask[i].astype(np.uint8),
# #                                      hooks=imgaug.HooksImages(activator=hook))

#             mask = det.augment_image(mask.astype(np.uint8),
#                                  hooks=imgaug.HooksImages(activator=hook))

            
#             if len(np.unique(mask)) != 2:
#                 return orig_image, target

#             # Verify that shapes didn't change
#             assert image.shape == image_shape, "Augmentation shouldn't change image size"
#             assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"

#         mask=np.transpose(mask, (2, 0, 1)) # num instance, h, w
#         bbox = extract_bboxes(mask)

#         target["boxes"] = torch.as_tensor(bbox, dtype=torch.float32)
#         target["masks"] = torch.as_tensor(mask.copy(), dtype=torch.uint8)

#         return image.copy(), target

import albumentations as A
import numpy as np
class AugTransforms(object):
    def __init__(self):
        self.train_transform = A.Compose([
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.1),
            A.RandomBrightnessContrast(p=0.1),
            A.Blur(p=0.1),
            A.RandomFog(p=0.1),
            A.RandomShadow(p=0.1),
            A.RandomSunFlare(p=0.1)

        ])

    def __call__(self, image, target):
        transformed = self.train_transform(image=np.asarray(image))
        image = transformed["image"]
        return image, target