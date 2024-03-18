import torch
import json
import numpy as np
import pandas as pd

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.modeling.roi_heads import CustomROIHeads
from detectron2.structures import Boxes, Instances, pairwise_iou
import shutil, os, glob, PIL
# import the packages for parsing the feature vectors
from detectron2.data import transforms as T
from detectron2.structures import ImageList
import cv2, sys
import argparse

# Define directories

# annotation = "./images/test/SS3/SS3_GT.json"    # Define the annotation file that we want to extract feature vectors for

# IMG_DIR = "./images/test/"                      # Define the image we want to extract feature vectors for
# is_pred = False                                 # True if extracting feature vectors from predicted boxes, False if from ground truth boxes


# # 1. Overwrite the directories and the config file
# def parse_args():
#     parser = argparse.ArgumentParser(description="Input the data parameters.")
#     parser.add_argument("--img_dir", default="../../images/test/", help="The directory of the images to extract feature vectors from.")
#     # parser.add_argument("--is_pred", default=False, choices=[True,False], help="Whether to extract feature vectors from predicted boxes or ground truth boxes.")
#     parser.add_argument("--is_pred", default=False, action=argparse.BooleanOptionalAction, help="Whether to extract feature vectors from predicted boxes or ground truth boxes.")
#     parser.add_argument("--annotation", default="../../images/test/SS4/SS4lim_GT.json", help="Annotation file to extract feature vectors from.")
#     parser.add_argument("--config_file", default="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml", help="The config file the feature vector extract relies on. Default ResNet-101.")
#     parser.add_argument("--model_path", default="model_230725limR101.pth", help="The path to the model to extract feature vectors from.")
#     args = parser.parse_args()
#     return args

# rewrite vector extraction into a class so that i can call the function from other files
class VectorExtraction:
    def __init__(self, 
                 out_dir="../../../SupContrast/output",
                 infer_dir: str = None, 
                 image_list: list = None, 
                 is_pred=False, 
                 annotation="../../images/test/SS4/SS4lim_GT.json", 
                 config_file="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
                 correct_config = True,
                 model_path="./models/model_230725limR101.pth",
                 num_classes=5):
        self.OUT_DIR = out_dir
        self.image_list = [os.path.join(infer_dir, os.path.basename(x)) for x in image_list]
        self.is_pred = is_pred
        self.annotation = annotation
        self.config_file = config_file
        self.model_path = model_path
        self.num_classes = num_classes
        if correct_config:
            self.cfg = self.load_config()
        else:
            self.cfg = config_file
            self.cfg.MODEL.WEIGHTS = self.model_path


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

    def load_config(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.config_file))
        cfg.INPUT.MIN_SIZE_TEST = 800
        cfg.INPUT.MAX_SIZE_TEST = 1333
        cfg.SOLVER.IMS_PER_BATCH = 8                                # This is the real "batch size" commonly known to deep learning people
        cfg.SOLVER.BASE_LR = 0.008                                  # pick a good LR 
        cfg.SOLVER.MAX_ITER = 30200                                 # iterations to train a practical dataset
        cfg.SOLVER.STEPS = []                                       # do not decay learning rate
        # cfg.SOLVER.CHECKPOINT_PERIOD = 1000                          # the number of iterations for every saved checkpoint
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512               # The "RoIHead batch size" is default to be 512. Scaled down to 128 for trial datasets.
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5                         # the number of classes
        # cfg.TEST.EVAL_PERIOD = 1000                                  # the number of steps interval to carry out an evaluation during training
        cfg.MODEL.WEIGHTS = self.model_path                               # path to the model we just trained
        # cfg.MODEL.WEIGHTS = os.path.join(OUTPUT_DIR, "model_SS3_comb.pth")                                     # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3                  # set a custom testing threshold
        return cfg

    def extract_vectors(self):
        # print(f"Extracting feature vectors from {self.model_path} using {self.config_file} for {self.annotation}...")

        # Add the rest of your vector extraction code here
        with open(self.annotation, 'r') as file:
            data = json.load(file)
            anno = data["annotations"]
            df = pd.DataFrame(anno)
            for i in range(len(df['bbox'])):
                x1, y1, w, h = df['bbox'][i]
                df['bbox'][i] = [x1, y1, float("{:.2f}".format(x1+w)), float("{:.2f}".format(y1+h))]            # [x1, y1, x2, y2] format
            
            img = data["images"]
            images_df = pd.DataFrame(img)

        
        # Method 2 of Parsing the Feature Vectors
        '''
        Via direct assignment of bbox
        Instead of guessing the best feature vector among the 1000 proposals for a bbox, maxpool the feature map by directly assigning the bbox.

        Skip the FC layer of predicting the class and the bbox filtering/refinement of the ROIHead (coz we know from the ground truth).

        Output the instance in the same format as from the RPN
        '''

        # Build model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictor = DefaultPredictor(self.cfg)

        for image_path in self.image_list:
            
            # try if the test image exists in the GT file (in case GT does not contain the image i.e. no annotations)
            try:
                image_id = int(images_df.loc[(images_df['file_name']==os.path.basename(image_path)), 'id'].values)
                # print(f"Image {os.path.basename(image_path)} ID {image_id} is included in GT" )
            except:
                continue

            # image_id = int(images_df.loc[(images_df['file_name']==os.path.basename(image_path)), 'id'].values)
            ground_truth_boxes = df[df['image_id']==image_id]['bbox'].to_list()
            ground_truth_classes = df[df['image_id']==image_id]['category_id'] - 1
            ground_truth_classes = ground_truth_classes.to_list()

            # Load the image and create Instances object for the ground truth boxes
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image, "height": height, "width": width}]

            # Use the predictor to get feature vectors for the ground truth boxes
            with torch.no_grad():
                images = predictor.model.preprocess_image(inputs)
                features = predictor.model.backbone(images.tensor)              # set of cnn features
                # proposal_generator = predictor.model.proposal_generator       # RPN not needed
                # proposals, _ = proposal_generator(images, features)  

                features_ = [features[f] for f in predictor.model.roi_heads.box_in_features]
                # box_features = predictor.model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])     # SKIPPED! box_feature directly parsed from GT
                boxes = [Boxes(torch.Tensor(ground_truth_boxes).to(device))]
                box_features = predictor.model.roi_heads.box_pooler(features_, boxes)
                box_features = predictor.model.roi_heads.box_head(box_features)                                             # feature vectors of the GT bboxes
                # print(box_features.shape)                                                                                 # the box_features are correct

                # Assemble the instances
                instances = Instances((height,width))
                instances.pred_boxes = Boxes(torch.tensor(ground_truth_boxes, dtype=torch.float32)).to(device)
                instances.scores = torch.tensor([[1] * len(ground_truth_boxes)][0], dtype=torch.float32).to(device)
                instances.pred_classes = torch.tensor(ground_truth_classes, dtype=torch.int).to(device)
                # instances.to(device)

                # SKIP the FC layer for classes and bbox filtering/refinement
                # predictions = predictor.model.roi_heads.box_predictor(box_features)             # logits of all fed in feature vectors (predictions[0].shape = [len(ground_truth_boxes),15])
                # pred_instances, pred_inds = predictor.model.roi_heads.box_predictor.inference(predictions, proposals)

                pred_instances = predictor.model.roi_heads.forward_with_given_boxes(features, [instances])

                # output boxes, masks, scores, etc
                pred_instances = predictor.model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
                # # features of the proposed boxes - SKIPPED!
                # feats = box_features[pred_inds]
                

            # Extract the feature vectors for each ground truth box and save them to a JSON file
            feature_dict = {}
            FOLDER_NAME = os.path.join(self.OUT_DIR,"embeddings",os.path.splitext(os.path.basename(image_path))[0])
            os.makedirs(FOLDER_NAME, exist_ok=True)
            for i, box in enumerate(ground_truth_boxes):
                # SKIP the IOU calculations
                # iou = pairwise_iou(Boxes(torch.as_tensor([box]).to(device)), proposals[0].proposal_boxes).squeeze()   
                # best_proposal_index = torch.argmax(iou)

                feature_vector = box_features[i].cpu().numpy().tolist()
                feature_dict[f"box_{i}"] = feature_vector
                
                # save vector
                if self.is_pred == True:
                    fv_json_start = "pred_feature_"
                else:
                    fv_json_start = "gt_feature_"
                FV_JSON = os.path.join(FOLDER_NAME,f"{fv_json_start}{i}.json")
                dict_print = {
                        "image_id": image_id,
                        "instance": i,
                        "bbox": box,
                        "category_id": ground_truth_classes[i],         # not +1 because codes later on reads from 0 to len-1
                        "vector": feature_vector
                    }
                with open(FV_JSON, "wt") as f:
                    json.dump(dict_print, f, cls=self.NpEncoder)

if __name__ == "__main__":
    
    vector_extractor = VectorExtraction()
    vector_extractor.extract_vectors()

    # # get parameters
    # args = parse_args()
    # IMG_DIR = args.img_dir
    # is_pred = args.is_pred
    # annotation = args.annotation
    # config_file = args.config_file
    # model_path = os.path.join(OUTPUT_DIR, args.model_path)
    


