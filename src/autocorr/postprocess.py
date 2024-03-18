# Functions for Goal 4 in merging masks and deleting redundant bboxes
import torch
import numpy as np
import cv2

# def mask_iou(mask1, mask2):
    # # Calculate the intersection and union of masks using logical AND and OR operation - OOM error
    # intersection = torch.logical_and(mask1, mask2)
    # union = torch.logical_or(mask1, mask2)

    # # Sum the intersection and union along the mask dimensions to get the areas
    # intersection_area = torch.sum(intersection, dim=(2, 3))
    # union_area = torch.sum(union, dim=(2, 3))

    # # Calculate the IoU for each pair of masks
    # iou = intersection_area / union_area

    # return iou

def mask_iou(masks):
    num_masks = len(masks)
    mask_ious = torch.zeros(num_masks, num_masks)

    # Calculate the intersection and union of masks using logical AND and OR operation
    for i in range(num_masks):
        for j in range(i+1, num_masks):
            # print("mask:", i, masks[i])
            # print("mask:", j, masks[j])
            intersection = torch.logical_and(masks[i], masks[j])
            union = torch.logical_or(masks[i], masks[j])
            iou = torch.sum(intersection, dim=(1, 2)) / torch.sum(union, dim=(1, 2))
            mask_ious[i, j] = iou
            mask_ious[j, i] = iou
    
    return mask_ious

def merge_masks(mask1, mask2):
    """
    Merge two masks.

    Args:
        mask1 (torch.Tensor): First mask of shape (H, W).
        mask2 (torch.Tensor): Second mask of shape (H, W).

    Returns:
        torch.Tensor: Merged mask if IoU is above the threshold, otherwise None.
    """
    
    merged_mask = torch.logical_or(mask1, mask2)
    return merged_mask

def find_remasked_bbox(mask, original_shape):
    """
    Find the bounding box that fits the resized mask.

    Args:
        mask (torch.Tensor): Resized mask of shape (H', W').
        original_shape (tuple): Shape of the original image or mask (H, W).

    Returns:
        tuple: Bounding box coordinates (x_min, y_min, x_max, y_max).
    """
    # Resize the mask to the original shape
    resized_mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=original_shape, mode='nearest')
    resized_mask = (resized_mask > 0.5).squeeze().cpu().numpy().astype(np.uint8)

    # Find the minimum and maximum coordinates that enclose the mask region
    rows = np.any(resized_mask, axis=1)
    cols = np.any(resized_mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return [x_min, y_min, x_max, y_max]

# Functions for the final outputs
def mask_to_polygon(maskedArr):
    """
    Convert a mask in a binary matrix of shape (H, W) to a polygon.
    Args:
        maskedArr (np.array): a binary matrix of shape (H,W) that contains True/False value
    Returns:
        segmentation[0] (np.array): an array of the polygon. [x1, y1, x2, y2...]
        area (float): the size of the polygon
        bbox (list): Bounding box coordinates in COCO format [x_min, y_min, delta_x, delta_y].
    """
    # Convert the binary mask to a binary image (0 and 255 values)
    binary_image = maskedArr.astype(np.uint8) * 255

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve the polygon coordinates for each contour
    polygons = []
    for contour in contours:
        if contour.size >= 6:
            polygons.append(contour.flatten().tolist())
    # Calculate areas and bbox
    area = np.sum(maskedArr)    
    bbox = find_bbox(maskedArr)

    return polygons, area, bbox

# also need to fix areas
def find_bbox(mask):
    """
    Find the bounding box that fits the resized mask.

    Args:
        mask (np.array): Resized mask of shape (H', W').
        original_shape (tuple): Shape of the original image or mask (H, W).

    Returns:
        list: Bounding box coordinates in COCO format [x_min, y_min, delta_x, delta_y].
    """
    # Find the minimum and maximum coordinates that enclose the mask region
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return [x_min, y_min, x_max-x_min, y_max-y_min]