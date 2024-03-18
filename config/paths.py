import datetime
import os

project_path = ""

# raw_data_path = os.path.join(project_path, "data/A14_L2")
# train_data_path = os.path.join(project_path, "data/A14_L2")
# val_data_path = os.path.join(project_path, "data/A14_L2")
# test_data_path = os.path.join(project_path, "data/A14_L2")

# raw_anns_filename = "data/annotations/A14_L2/raw.json"
# train_anns_filename = "data/annotations/A14_L2/train.json"
# val_anns_filename = "data/annotations/A14_L2/quick_test.json"
# test_anns_filename = "data/annotations/A14_L2/test.json"
# metadata_filename = "data/annotations/A14_L2/metadata.json"

raw_data_path = os.path.join(project_path, "images/train_all")
train_data_path = os.path.join(project_path, "images/train_all")
val_data_path = os.path.join(project_path, "images/train_all")
test_data_path = os.path.join(project_path, "images/train_all")

raw_anns_filename = "data/A12AL/A12AL_train4_00.json"
train_anns_filename = "data/A12AL/A12AL_train4_00.json"
val_anns_filename = "data/A12AL/A12AL_val4_SS4.json"
test_anns_filename = "data/A12AL/A12AL_rest4.json"
metadata_filename = "data/A12AL/metadata4.json"

raw_anns_path = os.path.join(project_path, raw_anns_filename)
train_anns_path = os.path.join(project_path, train_anns_filename)
val_anns_path = os.path.join(project_path, val_anns_filename)
test_anns_path = os.path.join(project_path, test_anns_filename)
metadata_path = os.path.join(project_path, metadata_filename)

output_path = os.path.join(project_path, "output/")
final_model_filename = "output"
final_model_path = os.path.join(project_path, "models")
final_model_full_path = os.path.join(project_path, "models/output.pth")
