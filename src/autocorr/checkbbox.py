# Functions for Goal 1 - check bbox movements/additions/deletions

def compare_bboxes(image_id, pseudolabels, ground_truths, ps_cat, gt_cat, centroid_threshold=50, areadiff_threshold=0.5):
    deleted_bboxes = []
    added_bboxes = []
    changed_size_bboxes = []

    pseudolabel_matched = [False] * len(pseudolabels)
    ground_truth_matched = [False] * len(ground_truths)

    # Check for deleted and changed size bounding boxes
    for idx, p_bbox in enumerate(pseudolabels):
        found_match = False
        for gt_idx, gt_bbox in enumerate(ground_truths):

            # find area difference
            p_area = (p_bbox[2] - p_bbox[0]) * (p_bbox[3] - p_bbox[1])
            gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
            area_diff = abs(p_area - gt_area) / max(p_area, gt_area)

            # find centroid distance
            p_centroid = ((p_bbox[0] + p_bbox[2]) / 2, (p_bbox[1] + p_bbox[3]) / 2)
            gt_centroid = ((gt_bbox[0] + gt_bbox[2]) / 2, (gt_bbox[1] + gt_bbox[3]) / 2)
            centroid_dist = ((p_centroid[0] - gt_centroid[0]) ** 2 + (p_centroid[1] - gt_centroid[1]) ** 2) ** 0.5

            # print(f'Psuedolabel {idx} vs GT label {gt_idx}, area diff = {area_diff}, centroid dist = {centroid_dist}')
            if area_diff <= areadiff_threshold and centroid_dist <= centroid_threshold:
                found_match = True
                ground_truth_matched[gt_idx] = True
                changed_size_bboxes.append((idx, p_bbox, image_id, ps_cat[idx]))
                break
            
            # # ICCCBE methods for comparison
            # iou_value = iou(p_bbox, gt_bbox)
            # if iou_value >= iou_threshold:
                # found_match = True
                # ground_truth_matched[gt_idx] = True
                # if abs(p_bbox[0] - gt_bbox[0]) > threshold or \
                #         abs(p_bbox[1] - gt_bbox[1]) > threshold or \
                #         abs(p_bbox[2] - gt_bbox[2]) > threshold or \
                #         abs(p_bbox[3] - gt_bbox[3]) > threshold:
                #     changed_size_bboxes.append((idx, p_bbox, image_id, ps_cat[idx]))
                # break

        if not found_match:
            deleted_bboxes.append((idx, p_bbox, image_id, ps_cat[idx]))

    # Check for added bounding boxes
    for idx, gt_bbox in enumerate(ground_truths):
        if not ground_truth_matched[idx]:
            added_bboxes.append((idx, gt_bbox, image_id, gt_cat[idx]))

    return deleted_bboxes, added_bboxes, changed_size_bboxes

def compare_cats(image_id, pseudolabels, ground_truths, ps_cat, gt_cat, iou_threshold=0.2):
    changed_cats_bboxes = []
    ground_truth_matched = [False] * len(ground_truths)
    
    # Check for deleted and changed size bounding boxes
    for idx, p_bbox in enumerate(pseudolabels):
        for gt_idx, gt_bbox in enumerate(ground_truths):
            iou_value = iou(p_bbox, gt_bbox)
            if iou_value >= iou_threshold:
                ground_truth_matched[gt_idx] = True
                if gt_cat[gt_idx] != ps_cat[idx]:       # changed GT categories
                    changed_cats_bboxes.append((idx, p_bbox, image_id, gt_cat[gt_idx]))
    
    return changed_cats_bboxes

def iou(bbox1, bbox2):
    # Calculate the Intersection over Union (IoU) of two bounding boxes
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    iou_value = intersection / (area_bbox1 + area_bbox2 - intersection)
    return iou_value