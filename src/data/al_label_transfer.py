import ast
import json
import os
import random

import numpy as np
import pandas as pd
from pycocotools.coco import COCO

from config import paths


def move_labels_to_train(
    image_list: list,
    output_path: str,
    output_file_tag: str = None,
    train_anns_file: str = None,
    test_anns_file: str = None,
):
    train_json_file = open(
        train_anns_file if train_anns_file is not None else paths.train_anns_path
    )
    train_json = json.load(train_json_file)

    test_json_file = open(
        test_anns_file if test_anns_file is not None else paths.test_anns_path
    )
    test_json = json.load(test_json_file)

    image_name_list = list()
    for img_name in image_list:
        image_name_list.append(img_name.split("/")[-1])

    image_list = image_name_list

    image_id_list = list()
    new_images = list()
    new_annotations = list()

    for img in test_json["images"]:
        if img["file_name"] in image_list:
            image_id_list.append(img["id"])
            new_images.append(img)
            test_json["images"].remove(img)

    for label in test_json["annotations"]:
        if label["image_id"] in image_id_list:
            new_annotations.append(label)
            test_json["annotations"].remove(label)

    train_json["images"].extend(new_images)
    train_json["annotations"].extend(new_annotations)

    os.makedirs(output_path, exist_ok=True)

    if output_file_tag is not None:
        with open(os.path.join(output_path, f"train_{str(output_file_tag).zfill(2)}.json"), "w") as f:
            json.dump(train_json, f)
        with open(os.path.join(output_path, f"test_{str(output_file_tag).zfill(2)}.json"), "w") as f:
            json.dump(test_json, f)
    else:
        with open(os.path.join(output_path, "train.json"), "w") as f:
            json.dump(train_json, f)
        with open(os.path.join(output_path, "test.json"), "w") as f:
            json.dump(test_json, f)

def trim_anno_to_newfile(
    image_list: list,
    out_dir: str,
    output_file_type: str = None,
    output_file_tag: str = None,
    test_anns_file: str = None,
):

    # to crop annotations from a big repo (the test set)
    test_json_file = open(
        test_anns_file if test_anns_file is not None else paths.test_anns_path
    )
    test_json = json.load(test_json_file)

    image_name_list = list()
    for img_name in image_list:
        image_name_list.append(img_name.split("/")[-1])

    image_list = image_name_list

    image_id_list = list()
    new_images = list()
    new_annotations = list()

    for img in test_json["images"]:
        if img["file_name"] in image_list:
            image_id_list.append(img["id"])
            new_images.append(img)

    for label in test_json["annotations"]:
        if label["image_id"] in image_id_list:
            new_annotations.append(label)

    dict_to_json = {
        "categories": test_json["categories"],
        "images": new_images,
        "annotations": new_annotations
    }

    os.makedirs(out_dir, exist_ok=True)

    if output_file_tag is not None:
        with open(os.path.join(out_dir, f"{output_file_type}_{str(output_file_tag).zfill(2)}.json"), "w") as f:
            json.dump(dict_to_json, f)
    else:
        with open(os.path.join(out_dir, f"{output_file_type}.json"), "w") as f:
            json.dump(dict_to_json, f)

def get_random_n_images(
    test_anns_file: str = None, 
    no_img: int = 10
    ):
    test_json_file = open(
        test_anns_file if test_anns_file is not None else paths.test_anns_path
    )
    test_json = json.load(test_json_file)

    image_list = list()
    rand_idx = random.sample(range(len(test_json["images"])), no_img)

    for idx in rand_idx:
        image_list.append(test_json["images"][idx]["file_name"])

    return image_list

def get_batch_images(test_anns_file: str = None):
    test_json_file = open(
        test_anns_file if test_anns_file is not None else paths.test_anns_path
    )
    test_json = json.load(test_json_file)
    img = pd.DataFrame(test_json["images"])
    image_list = img['file_name'].to_list()

    return image_list

# Move the AL/AutoCorr images plus more to the training set
def get_random_n_images_except_list(
    test_anns_file: str = None, 
    no_img: int = 10,
    except_list: list = None
    ):
    test_json_file = open(
        test_anns_file if test_anns_file is not None else paths.test_anns_path
    )
    test_json = json.load(test_json_file)
    for img in test_json["images"]:
        if img["file_name"] in except_list:
            test_json["images"].remove(img)
    
    image_list = list()
    rand_idx = random.sample(range(len(test_json["images"])), no_img)

    for idx in rand_idx:
        fname = os.path.basename(test_json["images"][idx]["file_name"])
        image_list.append(fname)

    return image_list

def move_alplus_to_train(
    image_list: list,                   # list of AL/AutoCorr image names
    output_path: str,
    output_file_tag: str = None,
    train_anns_file: str = None,
    test_anns_file: str = None,
    nos_extra: int = 20
):
    train_json_file = open(
        train_anns_file if train_anns_file is not None else paths.train_anns_path
    )
    train_json = json.load(train_json_file)

    test_json_file = open(
        test_anns_file if test_anns_file is not None else paths.test_anns_path
    )
    test_json = json.load(test_json_file)

    # The known AL list
    al_image_list = list()
    for img_name in image_list:
        al_image_list.append(img_name.split("/")[-1])
    # The new extra images picked from random
    extra_image_list = get_random_n_images_except_list(
        test_anns_file = test_anns_file, 
        no_img = nos_extra,
        except_list = image_list
        )
    # Combine the two lists
    image_list = al_image_list + extra_image_list

    image_id_list = list()
    new_images = list()
    new_annotations = list()

    for img in test_json["images"]:
        if img["file_name"] in image_list:
            image_id_list.append(img["id"])
            new_images.append(img)
            test_json["images"].remove(img)

    for label in test_json["annotations"]:
        if label["image_id"] in image_id_list:
            new_annotations.append(label)
            test_json["annotations"].remove(label)

    train_json["images"].extend(new_images)
    train_json["annotations"].extend(new_annotations)

    os.makedirs(output_path, exist_ok=True)

    if output_file_tag is not None:
        with open(os.path.join(output_path, f"train_{str(output_file_tag).zfill(2)}.json"), "w") as f:
            json.dump(train_json, f)
        with open(os.path.join(output_path, f"test_{str(output_file_tag).zfill(2)}.json"), "w") as f:
            json.dump(test_json, f)
    else:
        with open(os.path.join(output_path, "train.json"), "w") as f:
            json.dump(train_json, f)
        with open(os.path.join(output_path, "test.json"), "w") as f:
            json.dump(test_json, f)
