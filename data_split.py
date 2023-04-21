import json
import os

data_root = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/qianyinlong/challenge/AI-City-2023/Naturalistic-Driving-Action-Recognition/Data/Ori-data/A1"
with open(os.path.join(data_root, "data_split_0209.json"), "r") as fp:
    data = json.load(fp)

val_list = data["val"]
train_list = data["train"]

all_list = data["train"] + data["val"]

val_list_1 = train_list[:5]
train_list_1 = [item for item in all_list if item not in val_list_1]
data = {}
data["train"] = train_list_1
data["val"]= val_list_1
with open(os.path.join(data_root, "data_split_1.json"), "w") as fp:
    json.dump(data, fp)


val_list_2 = train_list[5:10]
train_list_2 = [item for item in all_list if item not in val_list_2]
data = {}
data["train"] = train_list_2
data["val"]= val_list_2
with open(os.path.join(data_root, "data_split_2.json"), "w") as fp:
    json.dump(data, fp)

val_list_3 = train_list[10:15]
train_list_3 = [item for item in all_list if item not in val_list_3]
data = {}
data["train"] = train_list_3
data["val"]= val_list_3
with open(os.path.join(data_root, "data_split_3.json"), "w") as fp:
    json.dump(data, fp)

val_list_4 = train_list[15:]
train_list_4 = [item for item in all_list if item not in val_list_4]
data = {}
data["train"] = train_list_4
data["val"]= val_list_4
with open(os.path.join(data_root, "data_split_4.json"), "w") as fp:
    json.dump(data, fp)