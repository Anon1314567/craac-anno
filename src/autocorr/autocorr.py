# # Add the config folder path to the sys.path list
import os, sys
sys.path.append('../../')
from config import config, paths, setup
from src.autocorr.vector_extraction import VectorExtraction
from src.autocorr.fvreject import CheckDiff, FvReject
from src.autocorr.evaluation import AutoCorrEval
from src.data import al_label_transfer

def extract(out_dir="../../../SupContrast/output", 
                 infer_dir="../../images/test/",
                 image_list=[], 
                 is_pred=False, 
                 gt_anno="../../images/test/SS4/SS4lim_GT.json", 
                 config_file="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
                 correct_config = True,
                 model_path="./models/model_230725limR101.pth",
                 num_classes=5):
    # Initialize the class with the necessary parameters
    vector_extractor = VectorExtraction(                
                 out_dir=out_dir, 
                 infer_dir=infer_dir,
                 image_list=image_list, 
                 is_pred=is_pred, 
                 annotation=gt_anno, 
                 config_file=config_file,
                 correct_config = correct_config,
                 model_path=model_path
                 )
    
    # Call the method to extract vectors
    vector_extractor.extract_vectors()

def correct(out_dir="../../output/embeddings",
            config_file=config.get_cfg_for_cns(**setup.exp_0528_cns_prediction),
            correct_config = False,
            model_path="model_231221lim.pth",
            batch = "SS3",
            gt_anno = "",
            corr_img_dir = "",
            pred_img_dir = "",
            infer_anno = "",
            end_anno = "",
            image_list = []
            ):
    checkdiff = CheckDiff(
        out_dir=out_dir,
        config_file=config_file,
        infer_dir=corr_img_dir,
        image_list=image_list, 
        annotation = gt_anno,
        centroid_tolerance=0.2, 
        areadiff_threshold=0.4)
    deleted_accum, added_accum, chcats_accum, deleted_vecs, added_vecs, chcats_vecs = checkdiff.check_diff()

    # try:
    #     iter = int(batch)
    #     if iter <= 10:
    #         sim_rm_thres = 0.9  # allow some deletions because predictions are bad
    #         sim_add_thres = 0.7
    #         sim_cat_thres = 0.7
    #         logit_thres = 0.04  # allow more additions
    #     else:
    #         # add only after 10 iterations
    #         sim_rm_thres = 1.0
    #         sim_add_thres = 0.7
    #         sim_cat_thres = 0.7
    #         logit_thres = 0.05
    # except ValueError:
    #     sim_rm_thres = 1.0
    #     sim_add_thres = 0.7
    #     sim_cat_thres = 0.7
    #     logit_thres = 0.07
    
    sim_rm_thres = 0.7
    sim_add_thres = 0.7
    sim_cat_thres = 0.7
    logit_thres = 0.04
    

    auto_corr = FvReject(out_dir=out_dir,
                 config_file=config_file,
                 correct_config = correct_config,
                 model_path=model_path,
                 num_classes=5,
                 batch = batch,
                 sim_rm_thres = sim_rm_thres,
                 sim_add_thres = sim_add_thres,
                 sim_cat_thres = sim_cat_thres,
                 logit_thres = logit_thres)
    # auto_corr.CorrAlike(deleted_vecs, deleted_accum, added_vecs, added_accum, chcats_vecs, chcats_accum,
    #             INFER_DIR = pred_img_dir,
    #             INFER_ANNO = infer_anno,
    #             END_ANNO = end_anno,
    #             image_list = [])
    auto_corr.CorrAlike2(deleted_vecs, deleted_accum, added_vecs, added_accum, chcats_vecs, chcats_accum,
            INFER_DIR = pred_img_dir,
            INFER_ANNO = infer_anno,
            END_ANNO = end_anno,
            image_list = [])
    # return deleted_accum, added_accum, chcats_accum, deleted_vecs, added_vecs, chcats_vecs

def infer(out_dir="../../output/embeddings",
            config_file=config.get_cfg_for_cns(**setup.exp_0528_cns_prediction),
            correct_config = False,
            model_path="model_231221lim.pth",
            batch = "SS3",
            infer_dir = "",
            infer_anno = "",
            end_anno = "",
            image_list = []):
    infer = FvReject(out_dir=out_dir,
                    config_file=config_file,
                    correct_config = correct_config,
                    model_path=model_path,
                    num_classes=5,
                    batch = batch,
                    sim_rm_thres = 1.0,
                    sim_add_thres = 0.7,
                    sim_cat_thres = 0.7)
    # infer.InferAsIs(INFER_DIR = infer_dir,
    #                 INFER_ANNO = infer_anno,
    #                 END_ANNO = end_anno,
    #                 image_list = image_list
    #                 )
    infer.CorrAlike2(
        INFER_DIR = infer_dir,
        INFER_ANNO = infer_anno,
        END_ANNO = end_anno,
        image_list = image_list)

def eval(out_dir="../../../SupContrast/output",
        gt_file = "",
        dt_file = "",
        end_eval_file = "",
        config_file="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"):
    eval = AutoCorrEval(out_dir=out_dir,
                        gt_file = gt_file,
                        dt_file = dt_file,
                        end_eval_file = end_eval_file,
                        config_file=config_file)
    eval.export_output()

def InferVal(it, 
        output_folder,
        model_weights, 
        config_file,
        infer_dir = ""):
    
    os.makedirs(f"{output_folder}/eval/", exist_ok=True)
    correct_config = False

    update_image_list = al_label_transfer.get_batch_images(
        test_anns_file=paths.val_anns_path
        )
    
    # make inference on the val set
    infer(
        out_dir = os.path.join(output_folder, "labels"),
        config_file = config_file,
        correct_config = correct_config,
        model_path = model_weights,
        batch = str(it).zfill(2),
        infer_dir = infer_dir,
        infer_anno = paths.metadata_path,
        end_anno = os.path.join(output_folder,f"labels/inferred_{str(it).zfill(2)}.json"),
        image_list = update_image_list
    )

    eval(out_dir=output_folder,
        gt_file = paths.val_anns_path,
        dt_file = os.path.join(output_folder,f"labels/inferred_{str(it).zfill(2)}.json"),
        end_eval_file = os.path.join(output_folder,f"eval/eval_infer_{str(it).zfill(2)}.xlsx"),
        config_file=config_file)
