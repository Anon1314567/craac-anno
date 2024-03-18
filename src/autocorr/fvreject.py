# Feature Vector Rejection

import os, glob, json
import torch
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, Instances, pairwise_iou, ImageList
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from collections import OrderedDict
import detectron2.data.transforms as T

# import from other codes I wrote
from src.autocorr.checkbbox import compare_bboxes, compare_cats, iou
from src.autocorr.simdecision import gen_vec, clusterDecision, sim_calc, breakAccum
from src.autocorr.postprocess import mask_iou, merge_masks, find_remasked_bbox, mask_to_polygon
from src.controller import d2_mask


# Goal 1 - Check differences between pseudo-labels and ground truth
class CheckDiff:
    def __init__(self, 
                 out_dir="../../../SupContrast/output",
                 config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                 infer_dir: str = None,                       
                 image_list: list = None,
                 annotation="../../images/test/SS4/SS4lim_GT.json",
                 centroid_tolerance=0.2, 
                 areadiff_threshold=0.4):
        self.image_list = [os.path.join(infer_dir, os.path.basename(x)) for x in image_list]
        # self.IMG_DIR = img_dir
        self.cfg = config_file
        self.annotation = annotation
        self.OUT_DIR = out_dir
        self.centroid_tolerance = centroid_tolerance        # % width deviation allowed from GT centroid
        self.areadiff_threshold = areadiff_threshold        
    
    # Add the rest of your vector extraction code here
    def check_diff(self):
        with open(self.annotation, 'r') as file:
            data = json.load(file)
            anno = data["annotations"]
            df = pd.DataFrame(anno)
            for i in range(len(df['bbox'])):
                x1, y1, w, h = df['bbox'][i]
                # change format without slicing
                df['bbox'][i] = [x1, y1, float("{:.2f}".format(x1+w)), float("{:.2f}".format(y1+h))]            # [x1, y1, x2, y2] format
                # print(df.head())
            img = data["images"]
            images_df = pd.DataFrame(img)

        image_list = [os.path.splitext(os.path.basename(x))[0] for x in self.image_list]
        # image_list = [os.path.splitext(os.path.basename(x))[0] for x in sorted(glob.glob(self.IMG_DIR+'/*.jpg'))]
        deleted_accum, added_accum, chcats_accum = [], [], []
        deleted_vecs, added_vecs, chcats_vecs = [], [], []
        
        logger, cfg = d2_mask.startup(regist_instances=False,cfg=self.cfg)

        embed_dir = os.path.join(self.OUT_DIR,"embeddings")

        for image in image_list:
            # print(f"reading image {image}")
            try:
                image_id = int(images_df.loc[(images_df['file_name']=="".join([image,".jpg"])), 'id'].values)
                dim = images_df.loc[(images_df['file_name']=="".join([image,".jpg"])), ['width', 'height']].values
                image_width = int(dim[0][0])
                image_height = int(dim[0][1])
            except:
                continue

            # read ground truth vectors
            gt_vecs, gt_bboxes, gt_catids = gen_vec(image,'gt_feature', embed_dir)
            pred_vecs, pred_bboxes, pred_catids = gen_vec(image,'pred_feature', embed_dir)

            # Function 1: check changes of bboxes
            # # ICCCBE version
            # deleted, added, changed_size = compare_bboxes(image_id, pred_bboxes, gt_bboxes, pred_catids, gt_catids, threshold=10, iou_threshold=0.5)
            logger.info(f"image id {image_id}")
            centroid_threshold = max(image_width, image_height) * self.centroid_tolerance
            deleted, added, changed_size = compare_bboxes(image_id, pred_bboxes, gt_bboxes, pred_catids, gt_catids, centroid_threshold, self.areadiff_threshold)
            changed_cats = compare_cats(image_id, pred_bboxes, gt_bboxes, pred_catids, gt_catids, iou_threshold=0.4)
            
            logger.info("Deleted bounding boxes:")
            for idx, bbox, im_id, cats in deleted:
                logger.info(f"Predicted Index: {idx}, BBox: {bbox}, del_category: {cats +1}")
            logger.info("Added bounding boxes:")
            for idx, bbox, im_id, cats in added:
                logger.info(f"Ground Truth Index: {idx}, BBox: {bbox}, add_category:, {cats +1}")
            logger.info("Bounding boxes with changed size:")
            for idx, bbox, im_id, cats in changed_size:
                logger.info(f"Predicted Index: {idx}, BBox: {bbox}, category: {cats+1}")
            logger.info("Bounding boxes with categories changed:")
            for idx, bbox, im_id, gt_cats in changed_cats:
                logger.info(f"Predicted Index: {idx}, BBox: {bbox}, new category: {gt_cats+1}")

            # Parse the vectors that are deleted or added
            if deleted:
                deleted_accum.append(deleted)
                deleted_vecs.append([pred_vecs[i] for i, v, im_id, cats in deleted])
            if added:
                added_accum.append(added)
                added_vecs.append([gt_vecs[i] for i, v, im_id, cats in added])
            if changed_cats:
                chcats_accum.append(changed_cats)
                chcats_vecs.append([pred_vecs[i] for i, v, im_id, gt_cats in changed_cats])
            
        return deleted_accum, added_accum, chcats_accum, deleted_vecs, added_vecs, chcats_vecs

class FvReject:
    def __init__(self, 
                 out_dir="../../../SupContrast/output",
                 config_file="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                 correct_config = True,
                 model_path="model_230725limR101.pth",
                 num_classes=5,
                 batch = "SS4",
                 sim_rm_thres = 0.9,
                 sim_add_thres = 0.8,
                 sim_cat_thres = 0.7,
                 logit_thres = 0.05):
        # config parameters
        self.OUT_DIR = out_dir
        self.config_file = config_file
        self.model_path = model_path
        self.num_classes = num_classes
        self.BATCH = batch
        if correct_config:
            self.cfg = self.load_config()
        else:
            self.cfg = config_file
            self.cfg.MODEL.WEIGHTS = self.model_path

        # fvreject inputs
        self.logit_thres = logit_thres   # the objectness logit thresholds to include proposal in calculation
        self.sim_rm_thres = sim_rm_thres
        self.sim_add_thres = sim_add_thres     # similarity thresholds (0.5 to add almost all cracks, 0.7 to get a precise enough result)
        self.sim_cat_thres = sim_cat_thres
        self.dim_rm_tol = 0.2     # the dimension tolerance to include proposal in calculation
        self.dim_add_tol = 0.2
        self.dim_cat_tol = 0.2
        self.iou_merge_thres = 0.4   # the iou threshold to merge two segmentation masks
        self.same_mask_thres = 0.95  # above the threshold, the merged mask will be considered to have the same area as the original mask
        self.augment_img = True
        self.vis_pred = False
        self.vis_proposal_bbox = False
        self.anno_pred = True
        # Set threshold of min area >= half of Q1 of each category from the full A12 annotation
        self.area_stat = {0: 14647, 1: 18163, 2: 2893, 3: 5469, 4: 9833} 
        # area_stat = {0: 32736, 1: 48069, 2: 14647, 3: 18163, 4: 24152, 5: 16773, 6: 34690, 9: 2893, 10: 5469, 11: 9833} 

    # Load config file
    def load_config(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.config_file))
        cfg.INPUT.MIN_SIZE_TEST = 800
        cfg.INPUT.MAX_SIZE_TEST = 1333
        cfg.SOLVER.IMS_PER_BATCH = 8                                # This is the real "batch size" commonly known to deep learning people
        cfg.SOLVER.BASE_LR = 0.008                                  # pick a good LR 
        cfg.SOLVER.MAX_ITER = 30200                                 # iterations to train a practical dataset
        cfg.SOLVER.STEPS = []                                       # do not decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512              # The "RoIHead batch size" is default to be 512. Scaled down to 128 for trial datasets.
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes          # the number of classes
        cfg.MODEL.WEIGHTS = self.model_path                         # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3                 # set a custom testing threshold
        return cfg
    
    def ClearDataset(self):
        # clear registered datasets
        dataset = 'fv_vis'
        DatasetCatalog.remove(dataset)
        MetadataCatalog.remove(dataset)

    # convert all np.integer, np.floating and np.ndarray into json recognisable int, float and lists
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    # Function of converting predictions of augmented images to the original resolution
    def original_res(self, ops, aug_instances, img, height, width):
        box = aug_instances[0].pred_boxes.tensor.cpu().numpy()
        dims = img['image'].cpu().numpy().shape
        ori_img = img['image'].cpu().numpy().reshape((dims[1],dims[2],dims[0]))
        input = T.AugInput(ori_img, boxes=box)
        if 'ResizeTransform' in str(ops):
            reaug = T.Resize(shape=(height, width))
        elif 'HFlipTransform' in str(ops):
            reaug = T.RandomFlip(prob=1, horizontal=True)
        else:
            return box
        transform = reaug(input)
        revised_boxes = input.boxes
        return revised_boxes

    def reshape_instance(self, ops, aug_instance, img, height, width):
        box = aug_instance.pred_boxes.tensor.cpu().numpy()
        dims = img['image'].cpu().numpy().shape
        ori_img = img['image'].cpu().numpy().reshape((dims[1],dims[2],dims[0]))
        input = T.AugInput(ori_img, boxes=box)
        if 'ResizeTransform' in str(ops):
            reaug = T.Resize(shape=(height, width))
            instance = detector_postprocess(aug_instance, height, width)
        elif 'HFlipTransform' in str(ops):
            reaug = T.RandomFlip(prob=1, horizontal=True)
            aug_instance.pred_boxes = Boxes(torch.tensor(input.boxes, dtype=torch.float32))
            instance = aug_instance
        else:
            instance = aug_instance
        # transform = reaug(input)
        # print(f"input: {input}")
        # print(f"transform: {transform}")
        # revised_boxes = input.boxes
        return instance

    # Function of making predictions of augmented images and outputting as an instance
    def AugPred(self, predictor, aug_inputs, augmented_transforms, height, width, device):
        aug_boxes, aug_scores, aug_classes = [], [], []
        pred_instances = Instances((height,width))
        
        # Prepare inputs for the predictor
        inputs = [{"image": img, "height": height, "width": width} for img in aug_inputs]
        original_image_size = [(height,width)] * len(aug_inputs)

        # Process all augmented images at once
        aug_img_forward = predictor.model.preprocess_image(aug_inputs)
        features = predictor.model.backbone(aug_img_forward.tensor)  # set of cnn features
        proposal_generator = predictor.model.proposal_generator
        proposals, _ = proposal_generator(aug_img_forward, features)  # RPN

        features_ = [features[f] for f in predictor.model.roi_heads.box_in_features]
        box_features = predictor.model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
        box_features = predictor.model.roi_heads.box_head(box_features)  # features of all 1k candidates
        predictions = predictor.model.roi_heads.box_predictor(box_features)  # logits of all 1k proposals (predictions[0].shape = [1000,15])
        aug_instances, aug_inds = predictor.model.roi_heads.box_predictor.inference(predictions, proposals)
        # clear masks if there are any - _postprocess is picky with masks
        for instance in aug_instances:
            if 'pred_masks' in instance._fields:
                instance.remove('pred_masks')
        
        # Postprocess predictions
        # Scale box to orig size
        aug_instances = predictor.model._postprocess(aug_instances, inputs, original_image_size)  
        # Apply inverse transformations to HFlipped image
        for i, (instances, transform) in enumerate(zip(aug_instances, augmented_transforms)):
            if 'HFlipTransform' in str(transform):
                if instances['instances'].has('pred_boxes'):
                    boxes = instances['instances'].pred_boxes.tensor.cpu().numpy()
                    boxes = transform.inverse().apply_box(boxes)
                    instances['instances'].pred_boxes = Boxes(torch.tensor(boxes, dtype=torch.float32).to(device))
                    aug_instances[i] = instances

        # Collect predictions from all augmented images
        for instances in aug_instances:
            for i in range(len(instances['instances'])):
                instance = instances['instances'][i]
                aug_boxes.append(instance.pred_boxes)
                aug_scores.append(instance.scores)
                aug_classes.append(instance.pred_classes)
        # skip image if no predictions
        if not aug_boxes: 
            return None 
              
        # combine predictions from all augmented images
        pred_instances.pred_boxes = Boxes(torch.stack([x.tensor[0] for x in aug_boxes]))
        pred_instances.scores = torch.cat(aug_scores)               # concatenate scores that are individual tensors
        pred_instances.pred_classes = torch.cat(aug_classes)
        pred_instances = [pred_instances]

        # # get predictions for each of the augmented images
        # # rewritten because many functions support batched inputs
        # for idx, img in enumerate(aug_inputs):
            
        #     # inputs = [{"image": img, "height": height, "width": width}]
        #     aug_img_forward = predictor.model.preprocess_image([img])
        #     features = predictor.model.backbone(aug_img_forward.tensor)  # set of cnn features
        #     proposal_generator = predictor.model.proposal_generator
        #     proposals, _ = proposal_generator(aug_img_forward, features)  # RPN

        #     features_ = [features[f] for f in predictor.model.roi_heads.box_in_features]
        #     box_features = predictor.model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
        #     box_features = predictor.model.roi_heads.box_head(box_features)                 # features of all 1k candidates
        #     predictions = predictor.model.roi_heads.box_predictor(box_features)             # logits of all 1k proposals (predictions[0].shape = [1000,15])
        #     aug_instances, aug_inds = predictor.model.roi_heads.box_predictor.inference(predictions, proposals)
        #     original_image_size = [(height,width)]
        #     # aug_instances = predictor.model._postprocess(aug_instances, inputs, original_image_size)  # scale box to orig size
        #     # print(aug_instances)

        #     # correct the pred_boxes back to original resolution
        #     aug_instances[0].pred_boxes = Boxes(torch.tensor(self.original_res(augmented_transforms[idx], aug_instances, img, height, width))).to(device)
        #     print(aug_instances)
        #     for i in range(len(aug_instances[0])):
        #         instance = aug_instances[0]
        #         aug_boxes.append(instance.pred_boxes[i])
        #         aug_scores.append(instance.scores[i])
        #         aug_classes.append(instance.pred_classes[i])


        # # skip image if no predictions
        # if not aug_boxes: 
        #     return None

        # # combine predictions from all augmented images
        # pred_instances.pred_boxes = Boxes(torch.stack([x.tensor[0] for x in aug_boxes]))
        # pred_instances.scores = torch.stack(aug_scores)
        # pred_instances.pred_classes = torch.stack(aug_classes)
        # pred_instances = [pred_instances]
        
        return pred_instances
    
    def takeAction(self,
                    pred_instance,
                    store_rm_dicts,
                    store_add_dicts,
                    store_cat_dicts,
                    box_vecs,
                    chcats_vecs,
                    chcats_accum,
                    pred_bboxes,
                    model,
                    device,
                    height,
                    width):
        pred_rm_bboxes = [x['pred_bbox'] for x in store_rm_dicts]
        # Get unique instances and their corresponding bboxes while preserving order
        unique_add_instances = set(x['pred_instance'] for x in store_add_dicts)
        pred_add_bboxes = [next(x['pred_bbox'] for x in store_add_dicts if x['pred_instance'] == instance) for instance in unique_add_instances]

        # Find categories for additions
        pred_add_class = []
        if pred_add_bboxes:
            pred_add_vecs_cpu = [box_vecs[x] for x in unique_add_instances]
            pred_add_vecs = torch.Tensor(pred_add_vecs_cpu).to(device)
            with torch.no_grad():
                pred_add_clogit = model.roi_heads.box_predictor(pred_add_vecs)
            # find index of max score
            for i, v in enumerate(pred_add_clogit[0]):
                value, add_class = torch.topk(v, 1)
                add_class = add_class[0]
                if add_class == len(v)-1:
                    value, add_class = torch.topk(v, 2)         # find the second largest if the most likely category is background
                    add_class = add_class[1]
                pred_add_class.append(add_class)
            # overwrite pred_add_class if the vector is similar to the ones in the change category bank
            if chcats_vecs:
                for i, chcats_vec in enumerate(chcats_vecs):
                    for j, pred_add_vec in enumerate(pred_add_vecs_cpu):
                        sim_add = sim_calc(chcats_vec, pred_add_vec)
                        # potentially have more than 1 vector in sim_add
                        sim_add_list = sim_add.tolist()
                        sim_id = [sim_add_list.index(x) for x in sim_add_list if x > self.sim_cat_thres]
                        for k in sim_id:
                            if sim_add[k] > self.sim_cat_thres:
                                _, cat_info = breakAccum(chcats_accum)
                                pred_add_class[j] = torch.tensor(cat_info[i][k][2]).to(device)

        filtered_boxes, filtered_scores, filtered_classes = [], [], []
        if pred_rm_bboxes or pred_add_bboxes:
            pred_rm_indices = [i for i, v1 in enumerate(pred_bboxes) for j, v2 in enumerate(pred_rm_bboxes) if iou(v1, v2) > 0.75]        
            # 75% IOU means length of intersection 93% of bbox, assuming all boxes are squares

            # Removal - pred_instances that are similar to the ones in the deleted bank
            for i in range(len(pred_instance)):
                if i not in pred_rm_indices:                    # including if pred_rm_bboxes = []
                    filtered_boxes.append(pred_instance.pred_boxes[i])
                    filtered_scores.append(pred_instance.scores[i])
                    # Change category - if the instance remains
                    if store_cat_dicts:
                        for x in store_cat_dicts:
                            if i == x['pred_instance']:
                                filtered_classes.append(torch.tensor(x['new_cat']).to(device))
                                break
                        else:
                            filtered_classes.append(pred_instance.pred_classes[i])
                    else:
                        filtered_classes.append(pred_instance.pred_classes[i])
            
            # Addition
            if pred_add_bboxes:
                for i in range(len(pred_add_class)):
                    filtered_boxes.append(Boxes(torch.tensor([pred_add_bboxes[i]], dtype=torch.float32)).to(device))
                    filtered_scores.append(torch.tensor(self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, dtype=torch.float32).to(device))
                    filtered_classes.append(pred_add_class[i])
            
        temp_instances = Instances((height,width)).to(device)
        if filtered_boxes:                                  # with pred_rm_bboxes or pred_add_bboxes
            temp_instances.pred_boxes = Boxes(torch.stack([x.tensor[0] for x in filtered_boxes]))
            temp_instances.scores = torch.stack(filtered_scores)
            temp_instances.pred_classes = torch.stack(filtered_classes)
            # print(temp_instances)
        elif pred_instance:                                 # if no changes
            temp_instances = pred_instance
        else:                                               # in case nothing in, nothing out
            temp_instances.pred_boxes = Boxes(torch.tensor([], dtype=torch.float32).to(device))
            temp_instances.scores = torch.tensor([], dtype=torch.float32).to(device)
            temp_instances.pred_classes = torch.tensor([], dtype=torch.int64).to(device)

        return temp_instances


    # Function to merge masks and eliminate unreasonably small masks
    def MergeElimMask(self, pred_instances, device):
        if isinstance(pred_instances[0], dict):
            # merge masks
            post_pred_bboxes = pred_instances[0]['instances'].pred_boxes
            post_pred_scores = pred_instances[0]['instances'].scores
            post_pred_masks = pred_instances[0]['instances'].pred_masks
            post_pred_catid = pred_instances[0]['instances'].pred_classes
        else:
            # merge masks
            post_pred_bboxes = pred_instances[0].pred_boxes
            post_pred_scores = pred_instances[0].scores
            post_pred_masks = pred_instances[0].pred_masks
            post_pred_catid = pred_instances[0].pred_classes
        # Reshape the masks tensor to have dimensions (N, 1, H, W)
        reshaped_masks = post_pred_masks.view(post_pred_masks.shape[0], 1, post_pred_masks.shape[1], post_pred_masks.shape[2])
        pred_masks_iou = mask_iou(reshaped_masks)

        # Loop through the IoU values for each mask pair
        if pred_masks_iou.shape:
            for i in range(pred_masks_iou.shape[0]):
                # print('mask {} of category {} with an area of {}'.format(i,post_pred_catid[i].item(),torch.sum(post_pred_masks[i]).item()))
                if torch.sum(post_pred_masks[i]).item()==0:
                    continue                                                                    # skip this mask if the mask is no longer present (e.g. deleted in previous iterations)
                elif pred_masks_iou.shape[0] == 1 or i == pred_masks_iou.shape[0]-1:            # for images with only one mask (hence can't iterate) or the last mask of the image
                    # Check 3: check if the area is too small
                    if post_pred_catid[i].item() in self.area_stat:
                        if torch.sum(post_pred_masks[i]).item() < self.area_stat[post_pred_catid[i].item()]:
                            mask_shape = (post_pred_masks.shape[1], post_pred_masks.shape[2])
                            post_pred_masks[i] = torch.zeros(mask_shape, dtype=torch.bool).to(device)       # Remove the i-th mask
                            # print(f'mask {i} was too small and hence deleted')
                else: 
                    for j in range(i + 1, pred_masks_iou.shape[1]):
                        # Check 1: check against the IOU threshold
                        # Check 2: check if the predicted mask is broadly the same as the merged mask
                        the_mask_iou = pred_masks_iou[i, j]
                        merged_mask = merge_masks(post_pred_masks[i], post_pred_masks[j])
                        same_mask = (torch.sum(post_pred_masks[i]) > self.same_mask_thres*torch.sum(merged_mask))
                        if (the_mask_iou > self.iou_merge_thres or same_mask == True) and post_pred_catid[i]==post_pred_catid[j]:
                            mask_shape = (post_pred_masks.shape[1], post_pred_masks.shape[2])
                            if post_pred_catid[i].item() in self.area_stat:
                                if torch.sum(merged_mask).item() > self.area_stat[post_pred_catid[i].item()]:
                                    post_pred_masks[i] = merged_mask                                                # Merge the i-th and j-th mask
                                else:
                                    post_pred_masks[i] = torch.zeros(mask_shape, dtype=torch.bool).to(device)       # Remove the i-th mask if the merged mask is too small
                            else:
                                post_pred_masks[i] = merged_mask                                                    # Merge the i-th and j-th mask
                            post_pred_masks[j] = torch.zeros(mask_shape, dtype=torch.bool).to(device)           # Clear the j-th mask
                            # print (f'mask {j} repeated with mask {i} and was hence deleted')
                            if torch.sum(post_pred_masks[i]) > 0:
                                post_pred_bboxes.tensor[i] = torch.tensor(find_remasked_bbox(post_pred_masks[i], mask_shape)).to(device)    # adjust the i-th bounding box
                        # Check 3: check if the area is too small
                        if post_pred_catid[i].item() in self.area_stat:
                            # print('post_pred_mask area {}. category {} requires an area of {}'.format(torch.sum(post_pred_masks[i]).item(), post_pred_catid[i].item(), area_stat[post_pred_catid[i].item()]))
                            if torch.sum(post_pred_masks[i]).item() < self.area_stat[post_pred_catid[i].item()]:
                                mask_shape = (post_pred_masks.shape[1], post_pred_masks.shape[2])
                                post_pred_masks[i] = torch.zeros(mask_shape, dtype=torch.bool).to(device)       # Remove the i-th mask
                                # print(f'mask {i} was too small and hence deleted')
        
        return post_pred_bboxes, post_pred_scores, post_pred_masks, post_pred_catid
    
    def OutputImage(self, 
                    image_path, 
                    fv_vis_metadata,
                    final_instances):

        img = cv2.imread(image_path)
        visualizer = Visualizer(img[:, :, ::-1], metadata=fv_vis_metadata, scale=1)

        # Removed the proposal_box function. Rarely used now anyway
        # if self.vis_proposal_bbox == True:
        #     # reconstruct proposal_instance
        #     proposal_bboxes_np = [np.array(boxes) for boxes in proposal_bboxes]
        #     proposal_bboxes_scores = [round(np.exp(v)/sum(np.exp(pbbox_logits)),2) for v in good_logits]
        #     proposal_bboxes_labels = ["".join(["#", str(i), " score", str(v)]) for i, v in enumerate(proposal_bboxes_scores)]
        #     out = visualizer.overlay_instances(masks=None, boxes=proposal_bboxes, labels=proposal_bboxes_labels)
        # else:
        if final_instances != 0:
            out = visualizer.draw_instance_predictions(final_instances[0]["instances"].to("cpu"))
        
        dst = f"{self.OUT_DIR}/img_{self.BATCH}/"
        os.makedirs(dst, exist_ok=True)
        
        save_name = "".join([dst, os.path.basename(image_path)])
        if final_instances != 0:
            cv2.imwrite(save_name, out.get_image()[:, :, ::-1])
        else:
            cv2.imwrite(save_name, img)
    
    def OutputAnno(self,
                   image_id,
                   image_path,
                   width,
                   height,
                   image_anno,
                   anno_anno,
                   defect_id,
                   final_instances,
                   INFER_ANNO,
                   END_ANNO):
        # Output predicted instances in annotation file
        image_anno.append({"id": image_id,
            "width": width,
            "height": height,
            "file_name": os.path.basename(image_path)})
        if final_instances != 0:
            anno_count = 0
            for x in range(len(final_instances[0]['instances'])):
                if sum(sum(final_instances[0]['instances'].pred_masks[x]))==0:
                    continue
                seg = mask_to_polygon(final_instances[0]['instances'].pred_masks[x].cpu().detach().numpy())
                if seg[0] == []:
                    continue
                else:
                    anno_dict = {"id": defect_id + anno_count,
                                "image_id": image_id,
                                "category_id": final_instances[0]['instances'].pred_classes[x].item() + 1,
                                "segmentation": seg[0],
                                "area": seg[1],
                                "bbox": seg[2],
                                "iscrowd": 0,
                                "attributes": {"occluded": False},
                                "score": round(final_instances[0]['instances'].scores[x].item(),4)}
                    anno_anno.append(anno_dict)
                    anno_count += 1
            defect_id = defect_id + anno_count
        
        return defect_id, image_anno, anno_anno


    # --------------------
    # New Function for inference with intervention
    # --------------------
                
    def CorrAlike2(self, 
                  deleted_vecs=None, 
                  deleted_accum=None, 
                  added_vecs=None,
                  added_accum=None,
                  chcats_vecs=None, 
                  chcats_accum=None,
                  INFER_DIR: str = None,
                  INFER_ANNO: str = None,
                  END_ANNO: str = None,
                  image_list: list = None):
        
        if image_list:
            image_list = [os.path.join(INFER_DIR, os.path.basename(x)) for x in image_list]       # for limited images
        else:
            image_list = sorted(glob.glob(INFER_DIR+'/*.jpg'))                  # for all images

        if self.vis_pred == True:
            register_coco_instances("fv_vis", {}, INFER_ANNO, INFER_DIR)
            with open(INFER_ANNO, 'r') as f:
                data = json.load(f)
            MetadataCatalog.get("fv_vis").set(thing_classes=[x['name'] for x in data['categories']])        # coerce to register 'thing_classes' in metadata
            fv_vis_metadata = MetadataCatalog.get('fv_vis')

        # Create dicts for anno_pred
        if self.anno_pred == True:
            defect_id = 1
            image_anno, anno_anno = [], []

        # -----------------------------
        # Goal 0 - Initialise model and set up image augmentation
        # -----------------------------

        logger, cfg = d2_mask.startup(regist_instances=False, cfg=self.cfg)

        # Build model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(self.cfg)
        model.eval()
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        # predictor = DefaultPredictor(self.cfg)    # try not to use this as it is not flexible enough

        # Define transformation
        augs = [
            T.ResizeShortestEdge(short_edge_length=(self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST), max_size=self.cfg.INPUT.MAX_SIZE_TEST),
            # T.RandomFlip(prob=1, horizontal=True),
            # T.RandomCrop(crop_type='relative', crop_size=(0.5, 0.5)),
            # T.RandomBrightness(intensity_min=0.6, intensity_max=1.4),
            # T.RandomContrast(intensity_min=0.6, intensity_max=1.4)
        ]

        
        for image_path in image_list:
            image_id = image_list.index(image_path) + 1     # can import from test.json if required. Not envisaged in real implementation however
            
            # Load the image to be inferred
            image = cv2.imread(image_path)
            height, width = image.shape[:2]

            # Initialize a list to store augmented versions of the image
            if self.augment_img == True:
                augmented_images = []
                augmented_transforms = []

                # Apply augmentations
                for t in augs:
                    ins = T.AugInput(image)
                    trnsfrm = t(ins)
                    aug_image = ins.image
                    augmented_images.append(aug_image)
                    augmented_transforms.append(trnsfrm)
                
                # Convert the augmented image to a tensor and create the input dictionary
                # "height" and "width" means the resolution of the desired OUTPUT. Set to the size of original resolution
                # See https://detectron2.readthedocs.io/en/v0.4.1/tutorials/models.html
                aug_inputs = [{"image": torch.as_tensor(aug_img.astype("float32").transpose(2, 0, 1)), "height": height, "width": width} for aug_img in augmented_images]
                # aug_inputs.append({"image": torch.as_tensor(image.astype("float32").transpose(2, 0, 1)), "height": height, "width": width})         # add the original at the back
                # augmented_transforms.append('original')
            else:
                aug_inputs = [{"image": torch.as_tensor(image.astype("float32").transpose(2, 0, 1)), "height": height, "width": width}]
                augmented_transforms = ['original']
            
            # -------------------------------
            # Goal 1a - do prediction on augmentation images
            # -------------------------------
            with torch.no_grad():

                logger.info(f"Start of Image: {image_id}")

                aug_boxes, aug_scores, aug_classes, aug_masks = [], [], [], []
                pred_instances = Instances((height,width))
            
                # Prepare inputs for the predictor
                # inputs = [{"image": img, "height": height, "width": width} for img in aug_inputs]
                original_image_size = [(height,width)] * len(aug_inputs)

                # Process all augmented images at once
                aug_img_forward = model.preprocess_image(aug_inputs)
                features = model.backbone(aug_img_forward.tensor)  # set of cnn features
                proposal_generator = model.proposal_generator
                proposals, _ = proposal_generator(aug_img_forward, features)  # RPN

                features_ = [features[f] for f in model.roi_heads.box_in_features]
                box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])   # features of all 1k candidates (torch.Size = [1000 x n,256,7,7])
                box_features = model.roi_heads.box_head(box_features)         # embeddings of all 1k candidates (torch.Size = [1000 x n,1024])
                predictions = model.roi_heads.box_predictor(box_features)     # logits of all 1k proposals (predictions[0].shape = [1000 x n,6])
                aug_instances, aug_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
                
                # Postprocess predictions
                for i in range(len(aug_inputs)):
                    transform = augmented_transforms[i]
                    if 'HFlipTransform' in str(transform):
                        if aug_instances[i].has('pred_boxes'):
                            boxes = aug_instances[i].pred_boxes.tensor.cpu().numpy()
                            boxes = transform.inverse().apply_box(boxes)
                            aug_instances[i].pred_boxes = Boxes(torch.tensor(boxes, dtype=torch.float32).to(device))
                    
                # Not postprocessing proposals here because the functions don't work with proposals
                # upscale the resized image after cluster actions before masking
                
                # operation by each image
                if (deleted_accum is not None) or (added_accum is not None) or (chcats_accum is not None):
                    for i in range(len(aug_inputs)):
                        # -------------------------
                        # Goal 1b - parse features for other steps
                        # -------------------------
                        # extract the right embeddings from the 1000 x n embeddings
                        start_ind = 1000 * i
                        end_ind = 1000 * (i+1)
                        # save vectors, bboxes for comparisons
                        # box_vecs for additions
                        all_vecs = box_features.cpu().numpy().tolist()[start_ind:end_ind]
                        proposal_bboxes = proposals[i].proposal_boxes.tensor.cpu().numpy().tolist()
                        pbbox_logits = proposals[i].objectness_logits.cpu().numpy().tolist()
                        # remove improbable bbox proposals to save time
                        good_logits = [v for i, v in enumerate(pbbox_logits) if np.exp(v)/sum(np.exp(pbbox_logits)) > self.logit_thres]    
                        proposal_bboxes = [proposal_bboxes[j] for j in range(len(good_logits))]
                        box_vecs = [all_vecs[j] for j in range(len(good_logits))]
                        # pred_vecs for deletions
                        # features of the proposed boxes
                        pred_index = aug_inds[i].cpu().numpy().tolist()
                        feats = [all_vecs[j] for j in pred_index]
                        pred_vecs = feats
                        pred_bboxes = aug_instances[i].pred_boxes.tensor.cpu().numpy().tolist()
                        pred_catid = aug_instances[i].pred_classes.cpu().numpy().tolist()
                        # print("box_vecs:", len(box_vecs))
                        # print("pred_vecs:", len(pred_vecs))

                        # -------------------------
                        # Goal 2 - clusterDecision
                        # -------------------------

                        # Version 2.0 - with clustering function
                        sim_thres = [self.sim_rm_thres, self.sim_add_thres, self.sim_cat_thres]
                        ratio_thres = [self.dim_rm_tol, self.dim_add_tol, self.dim_cat_tol]
                        # logger.info("Decision for pred_bboxes")
                        store_rm_dicts, store_cat_dicts = clusterDecision(image_id, deleted_vecs, deleted_accum, 
                                                                        added_vecs, added_accum,
                                                                        chcats_vecs, chcats_accum,
                                                                        pred_vecs, pred_bboxes, 
                                                                        sim_thres, ratio_thres, pred_catid=pred_catid)
                        # logger.info("Decision for proposal_bboxes")
                        store_add_dicts, _ = clusterDecision(image_id, deleted_vecs, deleted_accum, 
                                                        added_vecs, added_accum,
                                                        chcats_vecs, chcats_accum,
                                                        box_vecs, proposal_bboxes, 
                                                        sim_thres, ratio_thres)

                        # and use a case switch of actions to fit things back to store_rm_dicts, store_add_dicts, store_cat_dicts
                        if deleted_accum: logger.info(f"to remove: {store_rm_dicts}")
                        if added_accum: logger.info(f"to add: {store_add_dicts}")
                        if chcats_accum: logger.info(f"to change category: {store_cat_dicts}")

                        # -------------------------
                        # Goal 3 - take action
                        # -------------------------
                        
                        transform = augmented_transforms[i]
                        if 'ResizeTransform' in str(transform):
                            aug_height, aug_width = aug_inputs[i]['image'].shape[1], aug_inputs[i]['image'].shape[2]
                        else:
                            aug_height, aug_width = height, width
                        aug_instances[i] = self.takeAction(aug_instances[i],
                                                        store_rm_dicts,
                                                        store_add_dicts,
                                                        store_cat_dicts,
                                                        box_vecs,
                                                        chcats_vecs,
                                                        chcats_accum,
                                                        pred_bboxes,
                                                        model,
                                                        device,
                                                        aug_height,
                                                        aug_width)

                # Predict masks and upscale images
                # Predict masks - need to do it before concatenating instances from all augmented images
                # because features and aug_instances are recorded individually with the augmented images
                aug_instances = model.roi_heads.forward_with_given_boxes(features, aug_instances)
                # Upscale all augmented images 
                # dimension changes after using the detector_postprocess function so can't only upscale the resized image
                for i, instance in enumerate(aug_inputs):
                    if aug_instances[i].has('pred_boxes'):
                        aug_instances[i] = detector_postprocess(aug_instances[i], height, width)  # scale box to orig size
            
                # Collect predictions from all augmented images
                for instances in aug_instances:
                    for i in range(len(instances)):
                        instance = instances[i]
                        aug_boxes.append(instance.pred_boxes)
                        aug_scores.append(instance.scores)
                        aug_classes.append(instance.pred_classes)
                        aug_masks.append(instance.pred_masks)
                # combine predictions from all augmented images
                if aug_boxes:
                    pred_instances.pred_boxes = Boxes(torch.stack([x.tensor[0] for x in aug_boxes]))
                    pred_instances.scores = torch.cat(aug_scores)
                    pred_instances.pred_classes = torch.cat(aug_classes)
                    pred_instances.pred_masks = torch.cat(aug_masks)
                    pred_instances = [pred_instances]
                if not aug_boxes:
                    # skip image if no predictions
                    continue
                # print("combined:", pred_instances) # this is the place with pred_boxes, scores and pred_classes

                # -------------------------------
                # Goal 4 - Post prediction cleaning
                # -------------------------------
                # merge and eliminate small masks
                post_pred_bboxes, post_pred_scores, post_pred_masks, post_pred_catid = self.MergeElimMask(pred_instances, device)

                # Delete bboxes with empty masks
                final_mask_indices = [i for i, mask in enumerate(post_pred_masks) if torch.any(mask)]
                # print("End of Goal 4: ", final_mask_indices)
                if final_mask_indices:
                    final_instances = Instances((height,width))
                    final_instances.pred_boxes = Boxes(torch.stack([post_pred_bboxes.tensor[i].cpu() for i in final_mask_indices]))
                    final_instances.scores = torch.stack([post_pred_scores[i].cpu() for i in final_mask_indices])
                    final_instances.pred_classes = torch.stack([post_pred_catid[i].cpu() for i in final_mask_indices])
                    final_instances.pred_masks = torch.stack([post_pred_masks[i].cpu() for i in final_mask_indices])
                    final_instances = [{'instances': final_instances}]
                    # print(torch.sum(final_instances[0]['instances'].pred_masks), torch.sum(final_instances[0]['instances'].pred_classes))
                else:
                    final_instances = 0
                
                # print("final_instances:", final_instances)
            
                # Output annotated images
                if self.vis_pred == True:
                    self.OutputImage(
                        image_path, 
                        fv_vis_metadata,
                        final_instances)
                
                # Output predicted instances in annotation file
                if self.anno_pred == True:
                    defect_id, image_anno, anno_anno = self.OutputAnno(
                                                        image_id,
                                                        image_path,
                                                        width,
                                                        height,
                                                        image_anno,
                                                        anno_anno,
                                                        defect_id,
                                                        final_instances,
                                                        INFER_ANNO,
                                                        END_ANNO)
                   
        # clear registered datasets
        if self.vis_pred == True:
            DatasetCatalog.remove('fv_vis')
            MetadataCatalog.remove('fv_vis')
        
        if self.anno_pred==True:
            with open(INFER_ANNO, 'r') as file:
                data = json.load(file)
                category_anno = data["categories"]
                dict_to_json = {
                    "categories": category_anno,
                    "images": image_anno,
                    "annotations": anno_anno
                    }
                
            os.makedirs(f"{self.OUT_DIR}/labels/", exist_ok=True)

            with open(END_ANNO, "w") as outfile:
                json.dump(dict_to_json, outfile, cls=self.NpEncoder)

    # --------------------
    # Past Functions
    # --------------------

    def CorrAlike(self, 
                  deleted_vecs, deleted_accum, 
                  added_vecs, added_accum,
                  chcats_vecs, chcats_accum,
                  INFER_DIR: str = None,
                  INFER_ANNO: str = None,
                  END_ANNO: str = None,
                  image_list: list = None):
        
        if image_list:
            image_list = [os.path.join(INFER_DIR, os.path.basename(x)) for x in image_list]       # for limited images
        else:
            image_list = sorted(glob.glob(INFER_DIR+'/*.jpg'))                  # for all images

        if not INFER_DIR:
            INFER_DIR = "".join(['./images/test/',self.BATCH])
        if not INFER_ANNO:
            INFER_ANNO = "".join(['./images/test/',self.BATCH, "/", self.BATCH,"_test_lim.json"])
        if not END_ANNO:
            END_ANNO = "".join([self.OUT_DIR,"/",self.BATCH, "/", self.BATCH,"_coninferred.json"])

        if self.vis_pred == True:
            register_coco_instances("fv_vis", {}, INFER_ANNO, INFER_DIR)
            with open(INFER_ANNO, 'r') as f:
                data = json.load(f)
            MetadataCatalog.get("fv_vis").set(thing_classes=[x['name'] for x in data['categories']])        # coerce to register 'thing_classes' in metadata
            fv_vis_metadata = MetadataCatalog.get('fv_vis')

        # Create dicts for anno_pred
        if self.anno_pred == True:
            defect_id = 1
            image_anno, anno_anno = [], []

        # -----------------------------
        # Initialise model and set up image augmentation
        # -----------------------------

        logger, cfg = d2_mask.startup(regist_instances=False, cfg=self.cfg)

        # Build model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictor = DefaultPredictor(self.cfg)

        # Define transformation
        augs = [
            T.ResizeShortestEdge(short_edge_length=(self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST), max_size=self.cfg.INPUT.MAX_SIZE_TEST),
            T.RandomFlip(prob=1, horizontal=True),
            # T.RandomCrop(crop_type='relative', crop_size=(0.5, 0.5)),
            # T.RandomBrightness(intensity_min=0.6, intensity_max=1.4),
            # T.RandomContrast(intensity_min=0.6, intensity_max=1.4)
        ]

        
        for image_path in image_list:
            image_id = image_list.index(image_path) + 1     # can import from test.json if required. Not envisaged in real implementation however
            
            # Load the image to be inferred
            image = cv2.imread(image_path)
            height, width = image.shape[:2]

            # Initialize a list to store augmented versions of the image
            if self.augment_img == True:
                augmented_images = []
                augmented_transforms = []

                # Apply augmentations
                for t in augs:
                    ins = T.AugInput(image)
                    trnsfrm = t(ins)
                    aug_image = ins.image
                    augmented_images.append(aug_image)
                    augmented_transforms.append(trnsfrm)
                
                # Convert the augmented image to a tensor and create the input dictionary
                # "height" and "width" means the resolution of the desired OUTPUT. Set to the size of original resolution
                # See https://detectron2.readthedocs.io/en/v0.4.1/tutorials/models.html
                aug_inputs = [{"image": torch.as_tensor(aug_img.astype("float32").transpose(2, 0, 1)), "height": height, "width": width} for aug_img in augmented_images]
                aug_inputs.append({"image": torch.as_tensor(image.astype("float32").transpose(2, 0, 1)), "height": height, "width": width})         # add the original at the back
                augmented_transforms.append('original')
            
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            # Use the predictor to get feature vectors and proposal bboxes of the image to be inferred
            with torch.no_grad():
                # -------------------------------
                # Goal 2a - do prediction on augmentation images
                # -------------------------------
                
                logger.info(f"Start of Image: {image_id}")
                if self.augment_img == True:
                    aug_boxes, aug_scores, aug_classes = [], [], []
                    pred_instances = Instances((height,width))
                    
                    # get predictions for each of the augmented images
                    for idx, img in enumerate(aug_inputs):
                        aug_img_forward = predictor.model.preprocess_image([img])
                        features = predictor.model.backbone(aug_img_forward.tensor)  # set of cnn features
                        proposal_generator = predictor.model.proposal_generator
                        proposals, _ = proposal_generator(aug_img_forward, features)  # RPN

                        features_ = [features[f] for f in predictor.model.roi_heads.box_in_features]
                        box_features = predictor.model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
                        box_features = predictor.model.roi_heads.box_head(box_features)                 # features of all 1k candidates
                        predictions = predictor.model.roi_heads.box_predictor(box_features)             # logits of all 1k proposals (predictions[0].shape = [1000,15])
                        aug_instances, aug_inds = predictor.model.roi_heads.box_predictor.inference(predictions, proposals)

                        # correct the pred_boxes back to original resolution
                        aug_instances[0].pred_boxes = Boxes(torch.tensor(self.original_res(augmented_transforms[idx], aug_instances, img, height, width))).to(device)
                        for i in range(len(aug_instances[0])):
                            aug_boxes.append(aug_instances[0].pred_boxes[i])
                            aug_scores.append(aug_instances[0].scores[i])
                            aug_classes.append(aug_instances[0].pred_classes[i])

                    # combine predictions from all augmented images
                    if aug_boxes:
                        pred_instances.pred_boxes = Boxes(torch.stack([x.tensor[0] for x in aug_boxes]))
                        pred_instances.scores = torch.stack(aug_scores)
                        pred_instances.pred_classes = torch.stack(aug_classes)
                        pred_instances = [pred_instances]
                    # print("combined:", pred_instances) # this is the place with pred_boxes, scores and pred_classes

                # use the original image to generate features
                inputs = [{"image": image, "height": height, "width": width}]
                images = predictor.model.preprocess_image(inputs)
                features = predictor.model.backbone(images.tensor)  # set of cnn features
                features_ = [features[f] for f in predictor.model.roi_heads.box_in_features]

                # allow candidate proposals for additions
                proposal_generator = predictor.model.proposal_generator
                proposals, _ = proposal_generator(images, features)  # RPN
                box_features = predictor.model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
                box_features = predictor.model.roi_heads.box_head(box_features)                 # features of all 1k candidates
                predictions = predictor.model.roi_heads.box_predictor(box_features)             # logits of all 1k proposals (predictions[0].shape = [1000,15])
                if self.augment_img == True:
                    # for known predicted boxes - i.e. for deletions
                    if aug_boxes:
                        boxes = pred_instances[0].pred_boxes
                        box_features_known = predictor.model.roi_heads.box_pooler(features_, [boxes])
                        box_features_known = predictor.model.roi_heads.box_head(box_features_known)                         # feature vectors of the predicted bboxes
                        pred_inds = [i for i, x in enumerate(box_features_known)]                                           # we want all boxes that are fed into box_pooler
                    else:
                        pred_instances, pred_inds = predictor.model.roi_heads.box_predictor.inference(predictions, proposals)   # full run with RPN proposals if there were no predictions
                        box_features_known = box_features
                else:
                    # for unknown predicted boxes - i.e. for additions
                    pred_instances, pred_inds = predictor.model.roi_heads.box_predictor.inference(predictions, proposals)
                    box_features_known = box_features

                # save vectors, bboxes for comparisons
                # box_vecs for additions
                box_vecs = box_features.cpu().numpy().tolist()
                proposal_bboxes = proposals[0].proposal_boxes.tensor.cpu().numpy().tolist()
                pbbox_logits = proposals[0].objectness_logits.cpu().numpy().tolist()
                # remove too improbable bbox proposals to save time
                good_logits = [v for i, v in enumerate(pbbox_logits) if np.exp(v)/sum(np.exp(pbbox_logits)) > self.logit_thres]    
                proposal_bboxes = [proposal_bboxes[i] for i in range(len(good_logits))]
                box_vecs = [box_vecs[i] for i in range(len(good_logits))]
                # pred_vecs for deletions
                # features of the proposed boxes
                feats = box_features_known[pred_inds]
                pred_vecs = feats.cpu().numpy().tolist()
                pred_bboxes = pred_instances[0].pred_boxes.tensor.cpu().numpy().tolist()
                pred_catid = pred_instances[0].pred_classes.cpu().numpy().tolist()    
                # if pred_catid: print("original inference:", pred_instances[0].pred_classes)

            # ------------------------------
            # Goal 2 - compare the vectors
            # ------------------------------

            # Version 1.0 - ICCCBE
            # # adopted as below instead of a list comprehension to form a list per image, so that the shape matches the shape of deleted_vecs
            # if deleted_accum: del_bboxes, del_info = breakAccum(deleted_accum)
            # if added_accum: add_bboxes, add_info = breakAccum(added_accum)
            # if chcats_accum: cat_bboxes, cat_info = breakAccum(chcats_accum)

            # # remove predicted bboxes that are similar to the ones in the deleted bank
            # store_rm_dicts, store_add_dicts, store_cat_dicts = ([] for i in range(3))
            # if deleted_accum: store_rm_dicts = storeBank(image_id, deleted_vecs, del_bboxes, del_info, pred_vecs, pred_bboxes, sim_rm_thres, pred_cat=pred_catid)
            # if added_accum: store_add_dicts = storeBank(image_id, added_vecs, add_bboxes, add_info, box_vecs, proposal_bboxes, sim_add_thres)
            # if chcats_accum: store_cat_dicts = storeBank(image_id, chcats_vecs, cat_bboxes, cat_info, pred_vecs, pred_bboxes, sim_cat_thres, new_cat=True, pred_cat=pred_catid)

            # Version 2.0 - with clustering function
            sim_thres = [self.sim_rm_thres, self.sim_add_thres, self.sim_cat_thres]
            ratio_thres = [self.dim_rm_tol, self.dim_add_tol, self.dim_cat_tol]
            # logger.info("Decision for pred_bboxes")
            store_rm_dicts, store_cat_dicts = clusterDecision(image_id, deleted_vecs, deleted_accum, 
                                                            added_vecs, added_accum,
                                                            chcats_vecs, chcats_accum,
                                                            pred_vecs, pred_bboxes, 
                                                            sim_thres, ratio_thres, pred_catid=pred_catid)
            # logger.info("Decision for proposal_bboxes")
            store_add_dicts, _ = clusterDecision(image_id, deleted_vecs, deleted_accum, 
                                            added_vecs, added_accum,
                                            chcats_vecs, chcats_accum,
                                            box_vecs, proposal_bboxes, 
                                            sim_thres, ratio_thres)

            # and use a case switch of actions to fit things back to store_rm_dicts, store_add_dicts, store_cat_dicts
            if deleted_accum: logger.info(f"to remove: {store_rm_dicts}")
            if added_accum: logger.info(f"to add: {store_add_dicts}")
            if chcats_accum: logger.info(f"to change category: {store_cat_dicts}")

            # -------------------------------
            # Goal 3 - Take Action
            # -------------------------------
            pred_rm_bboxes = [x['pred_bbox'] for x in store_rm_dicts]
            # Get unique instances and their corresponding bboxes while preserving order
            unique_add_instances = set(x['pred_instance'] for x in store_add_dicts)
            pred_add_bboxes = [next(x['pred_bbox'] for x in store_add_dicts if x['pred_instance'] == instance) for instance in unique_add_instances]

            # Find categories for additions
            pred_add_class = []
            if pred_add_bboxes:
                pred_add_vecs_cpu = [box_vecs[x] for x in unique_add_instances]
                pred_add_vecs = torch.Tensor(pred_add_vecs_cpu).to(device)
                with torch.no_grad():
                    pred_add_clogit = predictor.model.roi_heads.box_predictor(pred_add_vecs)
                # find index of max score
                for i, v in enumerate(pred_add_clogit[0]):
                    value, add_class = torch.topk(v, 1)
                    add_class = add_class[0]
                    if add_class == len(v)-1:
                        value, add_class = torch.topk(v, 2)         # find the second largest if the most likely category is background
                        add_class = add_class[1]
                    pred_add_class.append(add_class)
                # overwrite pred_add_class if the vector is similar to the ones in the change category bank
                if chcats_vecs:
                    for i, chcats_vec in enumerate(chcats_vecs):
                        for j, pred_add_vec in enumerate(pred_add_vecs_cpu):
                            sim_add = sim_calc(chcats_vec, pred_add_vec)
                            # potentially have more than 1 vector in sim_add
                            sim_add_list = sim_add.tolist()
                            sim_id = [sim_add_list.index(x) for x in sim_add_list if x > self.sim_cat_thres]
                            for k in sim_id:
                                if sim_add[k] > self.sim_cat_thres:
                                    _, cat_info = breakAccum(chcats_accum)
                                    pred_add_class[j] = torch.tensor(cat_info[i][k][2]).to(device)


            if pred_rm_bboxes or pred_add_bboxes:
                pred_rm_indices = [i for i, v1 in enumerate(pred_bboxes) for j, v2 in enumerate(pred_rm_bboxes) if iou(v1, v2) > 0.75]        
                # 75% IOU means length of intersection 93% of bbox, assuming all boxes are squares

                # Removal - pred_instances that are similar to the ones in the deleted bank
                filtered_boxes, filtered_scores, filtered_classes = [], [], []
                for i in range(len(pred_instances[0])):
                    if i not in pred_rm_indices:                    # including if pred_rm_bboxes = []
                        filtered_boxes.append(pred_instances[0].pred_boxes[i])
                        filtered_scores.append(pred_instances[0].scores[i])
                        # Change category - if the instance remains
                        if store_cat_dicts:
                            for x in store_cat_dicts:
                                if i == x['pred_instance']:
                                    filtered_classes.append(torch.tensor(x['new_cat']).to(device))
                                    # print("pred_instance", i, "change added", x['new_cat'])
                                    break
                            else:
                                filtered_classes.append(pred_instances[0].pred_classes[i])
                                # print("pred_instance", i,"just added", pred_instances[0].pred_classes[i])
                        else:
                            filtered_classes.append(pred_instances[0].pred_classes[i])
                
                # Addition
                if pred_add_bboxes:
                    for i in range(len(pred_add_class)):
                        filtered_boxes.append(Boxes(torch.tensor([pred_add_bboxes[i]], dtype=torch.float32)).to(device))
                        filtered_scores.append(torch.tensor(self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, dtype=torch.float32).to(device))
                        filtered_classes.append(pred_add_class[i])
                
                temp_instances = Instances((height,width)).to(device)
                if filtered_boxes:
                    temp_instances.pred_boxes = Boxes(torch.stack([x.tensor[0] for x in filtered_boxes]))
                    temp_instances.scores = torch.stack(filtered_scores)
                    temp_instances.pred_classes = torch.stack(filtered_classes)
                    pred_instances = [temp_instances]
                else:                                               # in case after removal there are no bboxes left
                    continue

            else: pass

            # next you find masks and refinements etc..
            pred_instances = predictor.model.roi_heads.forward_with_given_boxes(features, pred_instances)                                          
            # output boxes, masks, scores, etc
            pred_instances = predictor.model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
            nos_instance = len(pred_instances[0]['instances'])
            # print(f'image {image_id} contains {nos_instance} instances in pred_instances')
            # print("end of Goal 3: bboxes", pred_instances[0]['instances'].pred_boxes)

            # -------------------------------
            # Goal 4 - Post prediction cleaning
            # -------------------------------
            # merge and eliminate small masks
            post_pred_bboxes, post_pred_scores, post_pred_masks, post_pred_catid = self.MergeElimMask(pred_instances, device)

            # Delete bboxes with empty masks
            final_mask_indices = [i for i, mask in enumerate(post_pred_masks) if torch.any(mask)]
            # print("End of Goal 4: ", final_mask_indices)
            if final_mask_indices:
                final_instances = Instances((height,width))
                final_instances.pred_boxes = Boxes(torch.stack([post_pred_bboxes.tensor[i].cpu() for i in final_mask_indices]))
                final_instances.scores = torch.stack([post_pred_scores[i].cpu() for i in final_mask_indices])
                final_instances.pred_classes = torch.stack([post_pred_catid[i].cpu() for i in final_mask_indices])
                final_instances.pred_masks = torch.stack([post_pred_masks[i].cpu() for i in final_mask_indices])
                final_instances = [{'instances': final_instances}]
                # print(torch.sum(final_instances[0]['instances'].pred_masks), torch.sum(final_instances[0]['instances'].pred_classes))
            else:
                final_instances = 0

            # -----------------------
            # Output annotated images
            # -----------------------
            if self.vis_pred == True:
                with open(INFER_ANNO, 'r') as file:
                    data_infer = json.load(file)

                img = cv2.imread(image_path)
                visualizer = Visualizer(img[:, :, ::-1], metadata=fv_vis_metadata, scale=1)

                if self.vis_proposal_bbox == True:
                    # reconstruct proposal_instance
                    proposal_bboxes_np = [np.array(boxes) for boxes in proposal_bboxes]
                    proposal_bboxes_scores = [round(np.exp(v)/sum(np.exp(pbbox_logits)),2) for v in good_logits]
                    proposal_bboxes_labels = ["".join(["#", str(i), " score", str(v)]) for i, v in enumerate(proposal_bboxes_scores)]
                    out = visualizer.overlay_instances(masks=None, boxes=proposal_bboxes, labels=proposal_bboxes_labels)
                else:
                    if final_instances != 0:
                        out = visualizer.draw_instance_predictions(final_instances[0]["instances"].to("cpu"))
                
                if not END_ANNO:
                    dst = "".join([self.OUT_DIR, "/", self.BATCH, "/"])
                    os.makedirs(dst, exist_ok=True)
                else:
                    dst = f"{self.OUT_DIR}/img_{self.batch}/"
                    os.makedirs(self.OUT_DIR, exist_ok=True)
                
                save_name = "".join([dst, os.path.basename(image_path)])
                if final_instances != 0:
                    cv2.imwrite(save_name, out.get_image()[:, :, ::-1])
                else:
                    cv2.imwrite(save_name, img)
            
            # ------------------------
            # Output predicted instances in annotation file
            # ------------------------
            if self.anno_pred == True:
                image_anno.append({"id": image_id,
                    "width": width,
                    "height": height,
                    "file_name": os.path.basename(image_path)})
                if final_instances != 0:
                    anno_count = 0
                    for x in range(len(final_instances[0]['instances'])):
                        seg = mask_to_polygon(final_instances[0]['instances'].pred_masks[x].cpu().detach().numpy())
                        if seg[0] == []:
                            continue
                        else:
                            anno_dict = {"id": defect_id + anno_count,
                                        "image_id": image_id,
                                        "category_id": final_instances[0]['instances'].pred_classes[x].item() + 1,
                                        "segmentation": seg[0],
                                        "area": seg[1],
                                        "bbox": seg[2],
                                        "iscrowd": 0,
                                        "attributes": {"occluded": False},
                                        "score": round(final_instances[0]['instances'].scores[x].item(),4)}
                            anno_anno.append(anno_dict)
                            anno_count += 1
                    defect_id = defect_id + anno_count

        # clear registered datasets
        if self.vis_pred == True:
            DatasetCatalog.remove('fv_vis')
            MetadataCatalog.remove('fv_vis')

        # output annotation file
        if self.anno_pred == True:
            with open(INFER_ANNO, 'r') as file:
                data = json.load(file)
                category_anno = data["categories"]
            dict_to_json = {
                "categories": category_anno,
                "images": image_anno,
                "annotations": anno_anno
                }
            
            if not END_ANNO:
                os.makedirs("".join([self.OUT_DIR,"/",self.BATCH, "/"]), exist_ok=True)
            else:
                os.makedirs(f"{self.OUT_DIR}/labels/", exist_ok=True)

            with open(END_ANNO, "w") as outfile:
                json.dump(dict_to_json, outfile, cls=self.NpEncoder)
    
    # --------------------
    # Inference without intervention
    # --------------------

    def InferAsIs(self,
                   INFER_DIR: str = None,
                   INFER_ANNO: str = None,
                   END_ANNO: str = None,
                   image_list: list = None):
        
        if image_list:
            image_list = [os.path.join(INFER_DIR, os.path.basename(x)) for x in image_list]
        else:
            image_list = sorted(glob.glob(INFER_DIR+'/*.jpg'))
        
        if not INFER_DIR:
            INFER_DIR = "".join(['./images/test/',self.BATCH])
        if not INFER_ANNO:
            INFER_ANNO = "".join(['./images/test/',self.BATCH, "/", self.BATCH,"_test_lim.json"])
        if not END_ANNO:
            END_ANNO = "".join([self.OUT_DIR,"/",self.BATCH, "/", self.BATCH,"_coninferred.json"])
    
        if self.vis_pred == True:
            register_coco_instances("fv_vis", {}, INFER_ANNO, INFER_DIR)
            with open(INFER_ANNO, 'r') as file:
                data_infer = json.load(file)
            MetadataCatalog.get("fv_vis").set(thing_classes=[x['name'] for x in data_infer['categories']])        # coerce to register 'thing_classes' in metadata
            fv_vis_metadata = MetadataCatalog.get('fv_vis')
        # Create dicts for anno_pred
        if self.anno_pred == True:
            defect_id = 1
            image_anno, anno_anno = [], []

        # Build model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictor = DefaultPredictor(self.cfg)

        # Define transformation
        augs = [
            T.ResizeShortestEdge(short_edge_length=(self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST), max_size=self.cfg.INPUT.MAX_SIZE_TEST),
            T.RandomFlip(prob=1, horizontal=True),
        ]
        
        for image_path in tqdm(image_list, desc="Running prediction on images"):
            image_id = image_list.index(image_path) + 1 # can import from test.json if required. Not envisaged in real implementation however
            
            # Load the image to be inferred
            image = cv2.imread(image_path)
            image = image[:, :, ::-1]           # change from RGB to BGR
            

            # Initialize a list to store augmented versions of the image
            augmented_images = []
            augmented_transforms = []

            # Apply augmentations
            for t in augs:
                ins = T.AugInput(image)
                trnsfrm = t(ins)
                aug_image = ins.image
                augmented_images.append(aug_image)
                augmented_transforms.append(trnsfrm)
            
            # Convert the augmented image to a tensor and create the input dictionary
            height, width = image.shape[:2]
            # "height" and "width" means the resolution of the desired OUTPUT. Set to the size of original resolution
            # See https://detectron2.readthedocs.io/en/v0.4.1/tutorials/models.html
            aug_inputs = [{"image": torch.as_tensor(aug_img.astype("float32").transpose(2, 0, 1)), "height": height, "width": width} for aug_img in augmented_images]
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            aug_inputs.append({"image": image, "height": height, "width": width})                   # add the original at the back
            augmented_transforms.append('original')
            

            # Use the predictor to get feature vectors and proposal bboxes of the image to be inferred
            with torch.no_grad():
                
                # get predictions for each of the augmented images
                pred_instances = self.AugPred(predictor, aug_inputs, augmented_transforms, height, width, device)
                # if AugPred returned None, skip this image and continue with the next one
                if pred_instances is None:
                    continue

                # use the original image to parse the needed box_features
                inputs = [{"image": image, "height": height, "width": width}]
                images = predictor.model.preprocess_image(inputs)
                features = predictor.model.backbone(images.tensor)  # set of cnn features
                features_ = [features[f] for f in predictor.model.roi_heads.box_in_features]
                boxes = pred_instances[0].pred_boxes
                box_features = predictor.model.roi_heads.box_pooler(features_, [boxes])
                box_features = predictor.model.roi_heads.box_head(box_features)                                             # feature vectors of the predicted bboxes
                # box_inds = torch.all(box_features==box_features, dim=1)
                box_inds = [i for i, x in enumerate(box_features)]
                
                # next you find masks and refinements etc..
                pred_instances = predictor.model.roi_heads.forward_with_given_boxes(features, pred_instances)                                         
                # output boxes, masks, scores, etc
                pred_instances = predictor.model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
                
            
                # -------------------------------
                # Goal 4 - Post prediction cleaning
                # -------------------------------
                # merge and eliminate small masks
                post_pred_bboxes, post_pred_scores, post_pred_masks, post_pred_catid = self.MergeElimMask(pred_instances, device)

                # Delete bboxes with empty masks
                final_mask_indices = [i for i, mask in enumerate(post_pred_masks) if torch.any(mask)]
                # print("End of Goal 4: ", final_mask_indices)
                if final_mask_indices:
                    final_instances = Instances((height,width))
                    final_instances.pred_boxes = Boxes(torch.stack([post_pred_bboxes.tensor[i].cpu() for i in final_mask_indices]))
                    final_instances.scores = torch.stack([post_pred_scores[i].cpu() for i in final_mask_indices])
                    final_instances.pred_classes = torch.stack([post_pred_catid[i].cpu() for i in final_mask_indices])
                    final_instances.pred_masks = torch.stack([post_pred_masks[i].cpu() for i in final_mask_indices])
                    final_instances = [{'instances': final_instances}]
                    # print(torch.sum(final_instances[0]['instances'].pred_masks), torch.sum(final_instances[0]['instances'].pred_classes))
                else:
                    final_instances = 0

            # -----------------------
            # Output annotated images
            # -----------------------
            if self.vis_pred == True:
                with open(INFER_ANNO, 'r') as file:
                    data = json.load(file)

                img = cv2.imread(image_path)
                visualizer = Visualizer(img[:, :, ::-1], metadata=fv_vis_metadata, scale=1)
                if final_instances != 0:
                    out = visualizer.draw_instance_predictions(final_instances[0]["instances"].to("cpu"))
                
                if not END_ANNO:
                    dst = "".join([self.OUT_DIR, "/", self.BATCH, "/"])
                    os.makedirs(dst, exist_ok=True)
                else:
                    dst = os.path.join(self.OUT_DIR, "img_", self.batch)
                    os.makedirs(self.OUT_DIR, exist_ok=True)
                    
                save_name = "".join([dst, os.path.basename(image_path)])
                if final_instances != 0:
                    cv2.imwrite(save_name, out.get_image()[:, :, ::-1])
                else:
                    cv2.imwrite(save_name, img)
            
            # ------------------------
            # Output predicted instances in annotation file
            # ------------------------
            if self.anno_pred == True:
                image_anno.append({"id": image_id,
                    "width": width,
                    "height": height,
                    "file_name": os.path.basename(image_path)})
                if final_instances != 0:
                    anno_count = 0
                    for x in range(len(final_instances[0]['instances'])):
                        if sum(sum(final_instances[0]['instances'].pred_masks[x]))==0:
                            continue
                        seg = mask_to_polygon(final_instances[0]['instances'].pred_masks[x].cpu().detach().numpy())
                        if seg[0] == []:
                            continue
                        else:
                            anno_dict = {"id": defect_id + anno_count,
                                        "image_id": image_id,
                                        "category_id": final_instances[0]['instances'].pred_classes[x].item() + 1,
                                        "segmentation": seg[0],
                                        "area": seg[1],
                                        "bbox": seg[2],
                                        "iscrowd": 0,
                                        "attributes": {"occluded": False},
                                        "score": round(final_instances[0]['instances'].scores[x].item(),4)}
                            anno_anno.append(anno_dict)
                            anno_count += 1
                    defect_id = defect_id + anno_count

        # clear registered datasets
        if self.vis_pred == True:
            DatasetCatalog.remove('fv_vis')
            MetadataCatalog.remove('fv_vis')

        # output annotation file
        if self.anno_pred == True:
            with open(INFER_ANNO, 'r') as file:
                data = json.load(file)
                category_anno = data["categories"]
            dict_to_json = {
                "categories": category_anno,
                "images": image_anno,
                "annotations": anno_anno
                }
            if not END_ANNO:
                os.makedirs("".join([self.OUT_DIR,"/",self.BATCH, "/"]), exist_ok=True)
            else:
                os.makedirs(f"{self.OUT_DIR}/labels/", exist_ok=True)
            
            with open(END_ANNO, "w") as outfile:
                json.dump(dict_to_json, outfile, cls=self.NpEncoder)


                