from datetime import datetime
import shutil
import sys
# Add the config folder path to the sys.path list
sys.path.append('../../')
from config import config, paths, setup
from ..controller import cas, d2_mask
from ..data import al_label_transfer
from ..scores import al_scoring
from ..autocorr import autocorr
import os
from detectron2.data.datasets import register_coco_instances


# -------------------------------
# 20240105 - My Experiments for CAS + Auto Correct
# -------------------------------
def inferEval(
        it, 
        output_folder,
        model_weights, 
        config_file
        ):
    
    os.makedirs(f"{output_folder}/eval/", exist_ok=True)
    correct_config = False

    update_image_list = al_label_transfer.get_batch_images(
        test_anns_file=paths.val_anns_path
        )
    
    # make inference on the val set
    autocorr.infer(
        out_dir = os.path.join(output_folder),
        config_file = config_file,
        correct_config = correct_config,
        model_path = model_weights,
        batch = str(it).zfill(2),
        infer_dir = "./images/train_all",
        infer_anno = paths.metadata_path,
        end_anno = os.path.join(output_folder,f"labels/Corr_{str(it).zfill(2)}.json"),
        image_list = update_image_list
    )

    autocorr.eval(out_dir=output_folder,
            gt_file = paths.val_anns_path,
            dt_file = os.path.join(output_folder,f"labels/Corr_{str(it).zfill(2)}.json"),
            end_eval_file = os.path.join(output_folder,f"eval/Corr_{str(it).zfill(2)}.xlsx"),
            config_file=config_file)

def AutocorrRoutine(
        it, 
        output_folder, 
        model_weights, 
        config_file,
        update_image_list: list = None): 

    if not update_image_list:

        update_image_list = al_scoring.get_bottom_n_images(
            score_file_path = f"./output/240212_craac/score_{str(it).zfill(2)}/al_score.csv", 
            no_img = 20,
        )


    # Check if config_file is a string or a cfg object
    if isinstance(config_file, str):
        correct_config = True
    else:
        correct_config = False


    
    # infer pseudolabels from trained model
    autocorr.infer(
        out_dir = output_folder,
        config_file = config_file,
        correct_config = correct_config,
        model_path = model_weights,
        batch = str(it).zfill(2),
        infer_dir = "./images/train_all",
        infer_anno = paths.metadata_path,
        # infer_anno = "./data/A12AL/metadata.json",
        end_anno = os.path.join(output_folder,f"labels/pl_{str(it).zfill(2)}.json"),
        image_list = update_image_list
    )

    # cut GT
    al_label_transfer.trim_anno_to_newfile(
        image_list=update_image_list,
        out_dir=os.path.join(output_folder, "labels"),
        output_file_type="gt",
        output_file_tag=str(it).zfill(2),
        test_anns_file = paths.test_anns_path
    )
    
    # extract feature vectors
    # clear the embeddings folder from previous runs
    shutil.rmtree(os.path.join(output_folder,"embeddings"), ignore_errors=True)  

    # Pseudolabel
    autocorr.extract(
        out_dir=output_folder, 
        infer_dir="./images/train_all",
        image_list=update_image_list,
        is_pred=True, 
        gt_anno=os.path.join(output_folder,f"labels/pl_{str(it).zfill(2)}.json"),
        # config_file="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        config_file = config_file, 
        correct_config = correct_config,
        model_path=model_weights
    )
    # Ground Truth
    autocorr.extract(
        out_dir=output_folder, 
        infer_dir="./images/train_all",
        image_list=update_image_list,
        is_pred=False, 
        gt_anno=os.path.join(output_folder,f"labels/gt_{str(it).zfill(2)}.json"), 
        # config_file = config.get_cfg_for_cns(**setup.exp_0528_cns_prediction),
        # config_file="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        config_file = config_file,
        correct_config = correct_config,
        model_path=model_weights
    )

    # check differences between PL and GT and carry out corrections
    autocorr.correct(
        out_dir=output_folder,
        config_file = config_file,
        correct_config = correct_config,
        model_path=model_weights,
        batch = str(it).zfill(2),
        gt_anno = os.path.join(output_folder,f"labels/gt_{str(it).zfill(2)}.json"),
        corr_img_dir = "./images/train_all",
        pred_img_dir = "./images/test/SS4",
        infer_anno = paths.metadata_path,
        end_anno = os.path.join(output_folder,f"labels/Corr_{str(it).zfill(2)}.json"),
        image_list = update_image_list
        )

    # # extract val set corrected pseudolabel
    # # may not be necessary because pred_img_dir above already confined to val set
    # val_image_list = al_label_transfer.get_batch_images(
    #     test_anns_file="./data/A12AL/A12AL_val_SS4.json"
    #     )
    # al_label_transfer.trim_anno_to_newfile(
    #     image_list=val_image_list,
    #     out_dir="./output/240105_cns/labels",
    #     output_file_type="coninferred",
    #     output_file_tag="00",
    #     test_anns_file = "./output/240105_cns/labels/test_00.json"
    # )

    if correct_config:
        register_coco_instances("test", {}, paths.val_anns_path, paths.raw_data_path)

    # evaluate the corrected PL
    os.makedirs(f"{output_folder}/eval/", exist_ok=True)

    autocorr.eval(out_dir=output_folder,
            gt_file = paths.val_anns_path,
            dt_file = os.path.join(output_folder,f"labels/Corr_{str(it).zfill(2)}.json"),
            end_eval_file = os.path.join(output_folder,f"eval/Corr_{str(it).zfill(2)}.xlsx"),
            config_file=config_file) 

def exp_240214_al(mode: str = "AL", iter: int = 10, train_from_init: bool = True):
    """
    Experiment on 14 Feb 24
    Comparison against full pipeline by using vanilla arch + random selection
    """
    logger, _ = d2_mask.startup(
        regist_instances=True, logfile=f"exp_240214_al_{datetime.now()}", cfg=None
    )

    if train_from_init:
        d2_mask.train_model_only(
            output_folder="240214_al/weights/init_model",
            regist_instances=False,
            cfg=config.get_cfg_for_al(**setup.exp_0528_al_model_init)
        )
        # d2_mask.get_coco_eval_results_al(
        #     model_weights=paths.final_model_full_path,
        #     regist_instances=False,
        #     output_path=f"./output/240214_al/init_model",
        #     cfg=config.get_cfg_for_al(**setup.exp_0528_al_prediction),
        # )
        autocorr.InferVal(it = 0, 
            output_folder='./output/240214_al',
            model_weights=paths.final_model_full_path, 
            config_file=config.get_cfg_for_al(**setup.exp_0528_al_prediction),
            infer_dir = "./images/train_all"
        )
        d2_mask.train_scores_only(
            output_folder="240214_al/weights/init_score",
            regist_instances=False,
            cfg=config.get_cfg_for_al(**setup.exp_0528_al_scores_init),
            model_weights=paths.final_model_full_path
        )

    if mode == "CAS" or mode == "AL":
        update_image_list = cas.sample_al_sets(
            model_weights=paths.final_model_full_path,
            regist_instances=False,
            output_path="./output/240214_al/init_score",
            test_anns_file=paths.test_anns_path,
            no_img=100,
        )
    else:
        update_image_list = cas.sample_rand_sets(
            test_anns_file=paths.test_anns_path,
            no_img=100,
        )

    cas.transfer_labels(
        train_anns_file=paths.train_anns_path,
        test_anns_file=paths.test_anns_path,
        image_list=update_image_list,
        output_path="./output/240214_al/labels",
        output_file_tag=0,
    )

    for it in range(int(iter)):
        cas.register_new_labels(
            train_anns_file=f"./output/240214_al/labels/train_{str(it).zfill(2)}.json",
            test_anns_file=f"./output/240214_al/labels/test_{str(it).zfill(2)}.json",
            iter_tag=it,
        )
        d2_mask.train_model_only(
            model_weights=paths.final_model_full_path,
            output_folder=f"240214_al/weights/model_{str(it).zfill(2)}",
            regist_instances=False,
            cfg=config.get_cfg_for_al(
                train_dataset=f"train_{str(it).zfill(2)}",
                **setup.exp_0528_al_model_cycle,
            ),
        )
        # d2_mask.get_coco_eval_results_al(
        #     model_weights=paths.final_model_full_path,
        #     regist_instances=False,
        #     output_path=f"./output/240214_al/model_{str(it).zfill(2)}",
        #     cfg=config.get_cfg_for_al(**setup.exp_0528_al_prediction),
        # )
        autocorr.InferVal(it = it+1, 
            output_folder='./output/240214_al',
            model_weights=paths.final_model_full_path, 
            config_file=config.get_cfg_for_al(**setup.exp_0528_al_prediction),
            infer_dir = "./images/train_all"
        )
        d2_mask.train_scores_only(
            model_weights=paths.final_model_full_path,
            output_folder=f"240214_al/weights/score_{str(it).zfill(2)}",
            regist_instances=False,
            cfg=config.get_cfg_for_al(
                train_dataset=f"train_{str(it).zfill(2)}", **setup.exp_0528_al_scores_cycle
            ),
        )

        if mode == "CAS" or mode == "AL":
            update_image_list = cas.sample_al_sets(
                model_weights=paths.final_model_full_path,
                regist_instances=False,
                output_path=f"./output/240214_al/score_{str(it).zfill(2)}",
                test_anns_file=f"./output/240214_al/labels/test_{str(it).zfill(2)}.json",
                no_img=100,
            )
        else:
            update_image_list = cas.sample_rand_sets(
                test_anns_file=f"./output/240214_al/labels/test_{str(it).zfill(2)}.json",
                no_img=100,
            )

        cas.transfer_labels(
            train_anns_file=f"./output/240214_al/labels/train_{str(it).zfill(2)}.json",
            test_anns_file=f"./output/240214_al/labels/test_{str(it).zfill(2)}.json",
            image_list=update_image_list,
            output_path="./output/240214_al/labels",
            output_file_tag=it + 1,
        )

def exp_240202_casac(mode: str = "CAS", iter: int = 10, train_from_init: bool = True, corrections: bool = True):
    """
    Experiment on 2 Feb 2024
    Testing the pipeline with CNS + AC + AL
    AL/AutoCorr Sample: 20, Step Train: 100
    """

    use_cns = True
    use_al = True

    if mode == "AL" or mode == "vanilla":
        use_cns = False

    if mode == "CNS" or mode == "vanilla":
        use_al = False

    logger, _ = d2_mask.startup(
        regist_instances=True, logfile=f"exp_{datetime.now()}", cfg=None
    )

    if train_from_init:
        cas.train_model(
            output_folder="240202_casac/weights/init_model",
            regist_instances=False,
            cfg=config.get_cfg_for_cns(**setup.exp_0528_cns_model_init),
        )
        cas.coco_eval(
            model_weights=paths.final_model_full_path,
            regist_instances=False,
            output_path="./output/240202_casac/init_model",
            cfg=config.get_cfg_for_cns(**setup.exp_0528_cns_prediction),
        )
        cas.train_score(
            output_folder="240202_casac/weights/init_score",
            regist_instances=False,
            cfg=config.get_cfg_for_cns(**setup.exp_0528_cns_scores_init),
            model_weights=paths.final_model_full_path
        )

    if mode == "CAS" or mode == "AL":
        # stop the validation set from being sampled
        restrict_list = al_label_transfer.get_batch_images(
            test_anns_file=paths.val_anns_path
        )
        update_image_list = cas.sample_al_sets(
            model_weights=paths.final_model_full_path,
            regist_instances=False,
            output_path="./output/240202_casac/init_score",
            test_anns_file=paths.test_anns_path,
            no_img=20,
            restrict_list = restrict_list
        )
    else:
        update_image_list = cas.sample_rand_sets(
            test_anns_file=paths.test_anns_path,
            no_img=20,
        )
        # update_image_list = al_label_transfer.get_batch_images(
        #     test_anns_file="./data/A12AL/A12AL_train_00.json"
        # )
    
    # Make predictions on the val set and evaluate
    autocorr.InferVal(it=0, 
        output_folder='./output/240202_casac',
        model_weights=paths.final_model_full_path, 
        config_file=config.get_cfg_for_cns(**setup.exp_0528_cns_prediction),
        infer_dir = "./images/train_all")
    
    # Auto-correction routine here...
    if corrections:
        AutocorrRoutine(
            it=0, 
            output_folder='./output/240202_casac', 
            model_weights=paths.final_model_full_path,
            config_file = config.get_cfg_for_cns(**setup.exp_0528_cns_prediction),
            update_image_list = update_image_list
        )

    al_label_transfer.move_alplus_to_train(
        image_list=update_image_list,                   # list of AL/AutoCorr image names
        output_path="./output/240202_casac/labels",
        output_file_tag=0,
        train_anns_file=paths.train_anns_path,
        test_anns_file=paths.test_anns_path,
        nos_extra=80
    )

    for it in range(int(iter)):            
        cas.register_new_labels(
            train_anns_file=f"./output/240202_casac/labels/train_{str(it).zfill(2)}.json",
            test_anns_file=f"./output/240202_casac/labels/test_{str(it).zfill(2)}.json",
            iter_tag=it,
        )
        cas.train_model(
            output_folder=f"240202_casac/weights/model_{str(it+1).zfill(2)}",
            regist_instances=False,
            cfg=config.get_cfg_for_cns(
                train_dataset=f"train_{str(it).zfill(2)}",
                unlabeled_dataset=f"test_{str(it).zfill(2)}",
                **setup.exp_0528_cns_model_cycle,
            ),
            model_weights=paths.final_model_full_path,
        )
        cas.coco_eval(
            model_weights=paths.final_model_full_path,
            regist_instances=False,
            output_path=f"./output/240202_casac/model_{str(it+1).zfill(2)}",
            cfg=config.get_cfg_for_cns(**setup.exp_0528_cns_prediction),
        )
        cas.train_score(
            output_folder=f"240202_casac/weights/score_{str(it+1).zfill(2)}",
            regist_instances=False,
            cfg=config.get_cfg_for_cns(
                train_dataset=f"train_{str(it).zfill(2)}",
                unlabeled_dataset=f"test_{str(it).zfill(2)}",
                **setup.exp_0528_cns_scores_cycle,
            ),
            model_weights=paths.final_model_full_path,
        )

        if mode == "CAS" or mode == "AL":
            
            restrict_list = al_label_transfer.get_batch_images(
                test_anns_file=paths.val_anns_path
            )

            update_image_list = cas.sample_al_sets(
                model_weights=paths.final_model_full_path,
                regist_instances=False,
                output_path=f"./output/240202_casac/score_{str(it+1).zfill(2)}",
                test_anns_file=f"./output/240202_casac/labels/test_{str(it).zfill(2)}.json",
                no_img=20,
                restrict_list = restrict_list
            )
        else:
            update_image_list = cas.sample_rand_sets(
                test_anns_file=f"./output/240202_casac/labels/test_{str(it).zfill(2)}.json",
                no_img=20,
            )

        # Make predictions on the val set and evaluate
        autocorr.InferVal(it=it+1, 
            output_folder='./output/240202_casac',
            model_weights=paths.final_model_full_path, 
            config_file=config.get_cfg_for_cns(**setup.exp_0528_cns_prediction),
            infer_dir = "./images/train_all")

        # Add pseudolabel routine here...
        if corrections:
            AutocorrRoutine(
                it=it + 1, 
                output_folder='./output/240202_casac', 
                model_weights=paths.final_model_full_path,
                config_file = config.get_cfg_for_cns(**setup.exp_0528_cns_prediction),
                update_image_list = update_image_list
            )

        al_label_transfer.move_alplus_to_train(
            image_list=update_image_list,                   # list of AL/AutoCorr image names
            output_path="./output/240202_casac/labels",
            output_file_tag=it + 1,
            train_anns_file=f"./output/240202_casac/labels/train_{str(it).zfill(2)}.json",
            test_anns_file=f"./output/240202_casac/labels/test_{str(it).zfill(2)}.json",
            nos_extra=80
        )

def exp_240204_casac(mode: str = "CAS", iter: int = 10, train_from_init: bool = True, corrections: bool = True):
    """
    Experiment on 4 Feb 2024
    Testing the pipeline with CNS + AC + AL
    Pick the bottom 20 images in the loss prediction (i.e. the most confident 20?)
    AL/AutoCorr Sample: 20, Step Train: 100
    """

    use_cns = True
    use_al = True

    if mode == "AL" or mode == "vanilla":
        use_cns = False

    if mode == "CNS" or mode == "vanilla":
        use_al = False

    logger, _ = d2_mask.startup(
        regist_instances=True, logfile=f"exp_{datetime.now()}", cfg=None
    )

    if train_from_init:
        cas.train_model(
            output_folder="240204_casac/weights/init_model",
            regist_instances=False,
            cfg=config.get_cfg_for_cns(**setup.exp_0528_cns_model_init),
        )
        cas.coco_eval(
            model_weights=paths.final_model_full_path,
            regist_instances=False,
            output_path="./output/240204_casac/init_model",
            cfg=config.get_cfg_for_cns(**setup.exp_0528_cns_prediction),
        )
        cas.train_score(
            output_folder="240204_casac/weights/init_score",
            regist_instances=False,
            cfg=config.get_cfg_for_cns(**setup.exp_0528_cns_scores_init),
            model_weights=paths.final_model_full_path
        )

    if mode == "CAS" or mode == "AL":
        # stop the validation set from being sampled
        restrict_list = al_label_transfer.get_batch_images(
            test_anns_file=paths.val_anns_path
        )
        update_image_list = cas.sample_alleast_sets(
            model_weights=paths.final_model_full_path,
            regist_instances=False,
            output_path="./output/240204_casac/init_score",
            test_anns_file=paths.test_anns_path,
            no_img=20,
            restrict_list = restrict_list
        )
    else:
        update_image_list = cas.sample_rand_sets(
            test_anns_file=paths.test_anns_path,
            no_img=20,
        )
        # update_image_list = al_label_transfer.get_batch_images(
        #     test_anns_file="./data/A12AL/A12AL_train_00.json"
        # )
    
    # Make predictions on the val set and evaluate
    autocorr.InferVal(it=0, 
        output_folder='./output/240204_casac',
        model_weights=paths.final_model_full_path, 
        config_file=config.get_cfg_for_cns(**setup.exp_0528_cns_prediction),
        infer_dir = "./images/train_all")
    
    # Auto-correction routine here...
    if corrections:
        AutocorrRoutine(
            it=0, 
            output_folder='./output/240204_casac', 
            model_weights=paths.final_model_full_path,
            config_file = config.get_cfg_for_cns(**setup.exp_0528_cns_prediction),
            update_image_list = update_image_list
        )

    al_label_transfer.move_alplus_to_train(
        image_list=update_image_list,                   # list of AL/AutoCorr image names
        output_path="./output/240204_casac/labels",
        output_file_tag=0,
        train_anns_file=paths.train_anns_path,
        test_anns_file=paths.test_anns_path,
        nos_extra=80
    )

    for it in range(int(iter)):            
        cas.register_new_labels(
            train_anns_file=f"./output/240204_casac/labels/train_{str(it).zfill(2)}.json",
            test_anns_file=f"./output/240204_casac/labels/test_{str(it).zfill(2)}.json",
            iter_tag=it,
        )
        cas.train_model(
            output_folder=f"240204_casac/weights/model_{str(it+1).zfill(2)}",
            regist_instances=False,
            cfg=config.get_cfg_for_cns(
                train_dataset=f"train_{str(it).zfill(2)}",
                unlabeled_dataset=f"test_{str(it).zfill(2)}",
                **setup.exp_0528_cns_model_cycle,
            ),
            model_weights=paths.final_model_full_path,
        )
        cas.coco_eval(
            model_weights=paths.final_model_full_path,
            regist_instances=False,
            output_path=f"./output/240204_casac/model_{str(it+1).zfill(2)}",
            cfg=config.get_cfg_for_cns(**setup.exp_0528_cns_prediction),
        )
        cas.train_score(
            output_folder=f"240204_casac/weights/score_{str(it+1).zfill(2)}",
            regist_instances=False,
            cfg=config.get_cfg_for_cns(
                train_dataset=f"train_{str(it).zfill(2)}",
                unlabeled_dataset=f"test_{str(it).zfill(2)}",
                **setup.exp_0528_cns_scores_cycle,
            ),
            model_weights=paths.final_model_full_path,
        )

        if mode == "CAS" or mode == "AL":
            
            restrict_list = al_label_transfer.get_batch_images(
                test_anns_file=paths.val_anns_path
            )

            update_image_list = cas.sample_alleast_sets(
                model_weights=paths.final_model_full_path,
                regist_instances=False,
                output_path=f"./output/240204_casac/score_{str(it+1).zfill(2)}",
                test_anns_file=f"./output/240204_casac/labels/test_{str(it).zfill(2)}.json",
                no_img=20,
                restrict_list = restrict_list
            )
        else:
            update_image_list = cas.sample_rand_sets(
                test_anns_file=f"./output/240204_casac/labels/test_{str(it).zfill(2)}.json",
                no_img=20,
            )

        # Make predictions on the val set and evaluate
        autocorr.InferVal(it=it+1, 
            output_folder='./output/240204_casac',
            model_weights=paths.final_model_full_path, 
            config_file=config.get_cfg_for_cns(**setup.exp_0528_cns_prediction),
            infer_dir = "./images/train_all")

        # Add pseudolabel routine here...
        if corrections:
            AutocorrRoutine(
                it=it + 1, 
                output_folder='./output/240204_casac', 
                model_weights=paths.final_model_full_path,
                config_file = config.get_cfg_for_cns(**setup.exp_0528_cns_prediction),
                update_image_list = update_image_list
            )

        al_label_transfer.move_alplus_to_train(
            image_list=update_image_list,                   # list of AL/AutoCorr image names
            output_path="./output/240204_casac/labels",
            output_file_tag=it + 1,
            train_anns_file=f"./output/240204_casac/labels/train_{str(it).zfill(2)}.json",
            test_anns_file=f"./output/240204_casac/labels/test_{str(it).zfill(2)}.json",
            nos_extra=80
        )

def exp_240212_vanilla(
    mode: str = "vanilla", iter: int = 10, train_from_init: bool = True
):
    """
    Experiment on 12 Feb 2024
    Comparison against full pipeline by using vanilla arch + random selection
    Same architecture as 240113_vanilla but different folder names + train set only with 4 categories
    """
    logger, _ = d2_mask.startup(
        regist_instances=True, logfile=f"exp_240212_vanilla_{datetime.now()}", cfg=None
    )

    if train_from_init:
        d2_mask.train_vanilla_mrcnn(
            output_folder="240212_vanilla/weights/init_model",
            regist_instances=False,
            cfg=config.get_cfg_for_vanilla(**setup.exp_0528_vanilla_model_init)
        )
        d2_mask.get_coco_eval_results_vanilla(
            model_weights=paths.final_model_full_path,
            regist_instances=False,
            output_path="./output/240212_vanilla/init_model",
            cfg=config.get_cfg_for_vanilla(**setup.exp_0528_vanilla_prediction),
        )
        autocorr.InferVal(it=0, 
            output_folder='./output/240212_vanilla',
            model_weights=paths.final_model_full_path, 
            config_file=config.get_cfg_for_vanilla(**setup.exp_0528_vanilla_prediction),
            infer_dir = "./images/train_all")

    update_image_list = cas.sample_rand_sets(
        test_anns_file=paths.test_anns_path,
        no_img=100,
    )

    cas.transfer_labels(
        train_anns_file=paths.train_anns_path,
        test_anns_file=paths.test_anns_path,
        image_list=update_image_list,
        output_path="./output/240212_vanilla/labels",
        output_file_tag=0,
    )

    for it in range(int(iter)):
        cas.register_new_labels(
            train_anns_file=f"./output/240212_vanilla/labels/train_{str(it).zfill(2)}.json",
            test_anns_file=f"./output/240212_vanilla/labels/test_{str(it).zfill(2)}.json",
            iter_tag=it,
        )
        d2_mask.train_vanilla_mrcnn(
            output_folder=f"240212_vanilla/weights/model_{str(it).zfill(2)}",
            regist_instances=False,
            cfg=config.get_cfg_for_vanilla(
                train_dataset=f"train_{str(it).zfill(2)}",
                **setup.exp_0528_vanilla_model_cycle,
            ),
            model_weights=paths.final_model_full_path,
        )
        d2_mask.get_coco_eval_results_vanilla(
            model_weights=paths.final_model_full_path,
            regist_instances=False,
            output_path=f"./output/240212_vanilla/model_{str(it).zfill(2)}",
            cfg=config.get_cfg_for_vanilla(**setup.exp_0528_vanilla_prediction),
        )

        autocorr.InferVal(it = it+1, 
            output_folder='./output/240212_vanilla',
            model_weights=paths.final_model_full_path, 
            config_file=config.get_cfg_for_vanilla(**setup.exp_0528_vanilla_prediction),
            infer_dir = "./images/train_all")

        update_image_list = cas.sample_rand_sets(
            test_anns_file=f"./output/240212_vanilla/labels/test_{str(it).zfill(2)}.json",
            no_img=100,
        )

        cas.transfer_labels(
            train_anns_file=f"./output/240212_vanilla/labels/train_{str(it).zfill(2)}.json",
            test_anns_file=f"./output/240212_vanilla/labels/test_{str(it).zfill(2)}.json",
            image_list=update_image_list,
            output_path="./output/240212_vanilla/labels",
            output_file_tag=it + 1,
        )

def exp_240212_craac(mode: str = "CAS", iter: int = 10, train_from_init: bool = True, corrections: bool = True):
    """
    Experiment on 12 Feb 2024
    Testing the pipeline with CNS + AC + AL
    AL/AutoCorr Sample: 20, Step Train: 100
    AL: pick the most uncertain 20 images for review
    AutoCorr: pick the least uncertain 20 images in the loss prediction that have predicted instances
    al_label_transfer = AL_20 + AutoCorr_20 + random 60
    """

    use_cns = True
    use_al = True

    if mode == "AL" or mode == "vanilla":
        use_cns = False

    if mode == "CNS" or mode == "vanilla":
        use_al = False

    logger, _ = d2_mask.startup(
        regist_instances=True, logfile=f"exp_{datetime.now()}", cfg=None
    )

    if train_from_init:
        cas.train_model(
            output_folder="240212_craac/weights/init_model",
            regist_instances=False,
            cfg=config.get_cfg_for_cns(**setup.exp_0528_cns_model_init),
        )
        cas.coco_eval(
            model_weights=paths.final_model_full_path,
            regist_instances=False,
            output_path="./output/240212_craac/init_model",
            cfg=config.get_cfg_for_cns(**setup.exp_0528_cns_prediction),
        )
        cas.train_score(
            output_folder="240212_craac/weights/init_score",
            regist_instances=False,
            cfg=config.get_cfg_for_cns(**setup.exp_0528_cns_scores_init),
            model_weights=paths.final_model_full_path
        )

    if mode == "CAS" or mode == "AL":
        # stop the validation set from being sampled
        restrict_list = al_label_transfer.get_batch_images(
            test_anns_file=paths.val_anns_path
        )
        update_image_list = cas.sample_al_sets(
            model_weights=paths.final_model_full_path,
            regist_instances=False,
            output_path="./output/240212_craac/init_score",
            test_anns_file=paths.test_anns_path,
            no_img=20,
            restrict_list = restrict_list
        )
        conf_image_list = al_scoring.get_bottom_n_images(
            score_file_path = f"./output/240212_craac/init_score/al_score.csv", 
            no_img = 20
        )

    else:
        update_image_list = cas.sample_rand_sets(
            test_anns_file=paths.test_anns_path,
            no_img=40,
        )
        conf_image_list = []
        # update_image_list = al_label_transfer.get_batch_images(
        #     test_anns_file="./data/A12AL/A12AL_train_00.json"
        # )
    
    # Make predictions on the val set and evaluate
    autocorr.InferVal(it=0, 
        output_folder='./output/240212_craac',
        model_weights=paths.final_model_full_path, 
        config_file=config.get_cfg_for_cns(**setup.exp_0528_cns_prediction),
        infer_dir = "./images/train_all")
    
    # Auto-correction routine here...
    if corrections:
        AutocorrRoutine(
            it=0, 
            output_folder='./output/240212_craac', 
            model_weights=paths.final_model_full_path,
            config_file = config.get_cfg_for_cns(**setup.exp_0528_cns_prediction),
            update_image_list = conf_image_list
        )

    want_image_list = update_image_list + conf_image_list
    al_label_transfer.move_alplus_to_train(
        image_list=want_image_list,                   # list of AL/AutoCorr image names
        output_path="./output/240212_craac/labels",
        output_file_tag=0,
        train_anns_file=paths.train_anns_path,
        test_anns_file=paths.test_anns_path,
        nos_extra=60
    )

    for it in range(int(iter)):            
        cas.register_new_labels(
            train_anns_file=f"./output/240212_craac/labels/train_{str(it).zfill(2)}.json",
            test_anns_file=f"./output/240212_craac/labels/test_{str(it).zfill(2)}.json",
            iter_tag=it,
        )
        cas.train_model(
            output_folder=f"240212_craac/weights/model_{str(it+1).zfill(2)}",
            regist_instances=False,
            cfg=config.get_cfg_for_cns(
                train_dataset=f"train_{str(it).zfill(2)}",
                unlabeled_dataset=f"test_{str(it).zfill(2)}",
                **setup.exp_0528_cns_model_cycle,
            ),
            model_weights=paths.final_model_full_path,
        )
        cas.coco_eval(
            model_weights=paths.final_model_full_path,
            regist_instances=False,
            output_path=f"./output/240212_craac/model_{str(it+1).zfill(2)}",
            cfg=config.get_cfg_for_cns(**setup.exp_0528_cns_prediction),
        )
        cas.train_score(
            output_folder=f"240212_craac/weights/score_{str(it+1).zfill(2)}",
            regist_instances=False,
            cfg=config.get_cfg_for_cns(
                train_dataset=f"train_{str(it).zfill(2)}",
                unlabeled_dataset=f"test_{str(it).zfill(2)}",
                **setup.exp_0528_cns_scores_cycle,
            ),
            model_weights=paths.final_model_full_path,
        )

        if mode == "CAS" or mode == "AL":
            
            restrict_list = al_label_transfer.get_batch_images(
                test_anns_file=paths.val_anns_path
            )

            update_image_list = cas.sample_al_sets(
                model_weights=paths.final_model_full_path,
                regist_instances=False,
                output_path=f"./output/240212_craac/score_{str(it+1).zfill(2)}",
                test_anns_file=f"./output/240212_craac/labels/test_{str(it).zfill(2)}.json",
                no_img=20,
                restrict_list = restrict_list
            )
            conf_image_list = al_scoring.get_bottom_n_images(
                score_file_path = f"./output/240212_craac/score_{str(it+1).zfill(2)}/al_score.csv", 
                no_img = 20
            )
        else:
            update_image_list = cas.sample_rand_sets(
                test_anns_file=f"./output/240212_craac/labels/test_{str(it).zfill(2)}.json",
                no_img=40,
            )

        # Make predictions on the val set and evaluate
        autocorr.InferVal(it=it+1, 
            output_folder='./output/240212_craac',
            model_weights=paths.final_model_full_path, 
            config_file=config.get_cfg_for_cns(**setup.exp_0528_cns_prediction),
            infer_dir = "./images/train_all")

        # Add pseudolabel routine here...
        if corrections:
            AutocorrRoutine(
                it=it + 1, 
                output_folder='./output/240212_craac', 
                model_weights=paths.final_model_full_path,
                config_file = config.get_cfg_for_cns(**setup.exp_0528_cns_prediction),
                update_image_list = conf_image_list
            )

        want_image_list = update_image_list + conf_image_list
        al_label_transfer.move_alplus_to_train(
            image_list=want_image_list,                   # list of AL/AutoCorr image names
            output_path="./output/240212_craac/labels",
            output_file_tag=it + 1,
            train_anns_file=f"./output/240212_craac/labels/train_{str(it).zfill(2)}.json",
            test_anns_file=f"./output/240212_craac/labels/test_{str(it).zfill(2)}.json",
            nos_extra=60
        )
