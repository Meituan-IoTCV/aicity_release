#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import pickle
import torch
import tqdm

import os 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import numpy as np
import matplotlib.pyplot as plt
from tools import util_eval
import re


dash_w = np.array([0.29952133,0.33819003,0.22717105,0.37327538,0.46176838,0.13216463
,0.16151206,0.08160829,0.25061411,0.27307249,0.34101561,0.347911
,0.32329934,0.60937417,0.37407451,0.19476108])
rear_w = np.array([0.30443487,0.18982168,0.55915577,0.42856789,0.30421331,0.53734027
,0.38533537,0.47688041,0.33448637,0.52617198,0.40130076,0.29074854
,0.33584091,0.3155264,0.51392694,0.3958831])
right_w  = np.array([0.3960438,0.47198829,0.21367318,0.19815673,0.23401831,0.3304951
,0.45315258,0.44151129,0.41489952,0.20075553,0.25768363,0.36134046
,0.34085974,0.07509943,0.11199855,0.40935582])

def plot_probs(probs, video_id):
    class_names = [
    "Normal",
    "Drinking",
    "Phone Call（Right）",
    "Phone Call（Left）",
    "Eating",
    "Text(Right)",
    "Text(Left)",
    "Reaching behind",
    "Adjust Control Pane",
    "Pick up from floor(Driver)",
    "Pick up from floor(Passenger)",
    "Talk to passenger at the right",
    "Talk to passenger at the backseat",
    "yawning",
    "Hand on head",
    "Singing or dance with music"
    ]
    NUM_CLASSES = probs.shape[1]
    seq_len = probs.shape[0]
    fig = plt.figure(figsize=(24, 24))
    for i in range(NUM_CLASSES):
        ax = fig.add_subplot(NUM_CLASSES, 1, i+1)
        ax.plot(list(range(seq_len)), probs[:, i])
        ax.set_xlabel("{}".format(class_names[i]))

    plt.tight_layout()
    plt.savefig("{}_probs.png".format(video_id), bbox_inches="tight")


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]

cmap = np.asarray(label_colours) / 255.0


def process_overlap(data):
    data = data.sort_values(by=["start"]).reset_index(drop=True)
    for j in range(len(data)-1):
        # 两个框重合
        if data.loc[j+1, "start"] - data.loc[j, "end"] <= 2:
            label_a = int(data.loc[j, "label"])
            label_b = int(data.loc[j+1, "label"])

            # 重合框为打电话、发短信，发短信作为打电话准备动作与打电话合并
            if (label_a in [2, 3, 5, 6]) and ( label_b in [2, 3, 5, 6]):
                if label_a in [2,3] and label_b in [2,3]:
                    continue
                if label_a in [5,6] and label_b in [5,6]:
                    continue
                # label_b = min(label_a, label_b)
                # data.loc[j+1, "start"] = min(data.loc[j, "start"], data.loc[j+1, "start"])
                # data.loc[j+1, "end"] = max(data.loc[j, "end"], data.loc[j+1, "end"])
                # data.loc[j+1, "label"] = label_b
                # data.loc[j, "end"] = 0
                # data.loc[j, "start"] = 0
                # 如果有发短信在打电话之前，直接删掉发短信的 segments
                if label_a in [2, 3] and label_b in [5, 6]:
                    data.loc[j+1, "end"] = 0
                    data.loc[j+1, "start"] = 0
                if label_a in [5, 6] and label_b in [2, 3]:
                    data.loc[j, "start"] = 0
                    data.loc[j, "end"] = 0
            # # 重合框中有一个时间比较短 <=3s, 直接短区间直接合并
            elif (data.loc[j+1, "end"] - data.loc[j+1, "start"]) > (data.loc[j, "end"] - data.loc[j, "start"]):
                if max(data.loc[j, "end"], data.loc[j+1, "end"]) - min(data.loc[j, "start"], data.loc[j+1, "start"]) > 20:
                    continue
                data.loc[j+1, "start"] =  min(data.loc[j, "start"], data.loc[j+1, "start"])
                data.loc[j+1, "end"] = max(data.loc[j, "end"], data.loc[j+1, "end"])
                data.loc[j+1, "label"] = label_b
                data.loc[j, "end"] = 0
                data.loc[j, "start"] = 0
            elif (data.loc[j+1, "end"] - data.loc[j+1, "start"]) < (data.loc[j, "end"] - data.loc[j, "start"]):
                if max(data.loc[j, "end"], data.loc[j+1, "end"]) - min(data.loc[j, "start"], data.loc[j+1, "start"]) > 20:
                    continue
                data.loc[j+1, "start"] = min(data.loc[j, "start"],  data.loc[j+1, "start"])
                data.loc[j+1, "end"] = max(data.loc[j, "end"], data.loc[j+1, "end"])               
                data.loc[j+1, "label"] = label_a
                data.loc[j, "end"] = 0
                data.loc[j, "start"] = 0  
    data = data[data["end"]!=0]            
    return data 

def merge_and_remove(data, merge_threshold=16):
    """
    对于同类别动作，如果间隔小于等于 16s， 则合并
    """
    df_total = pd.DataFrame([[0, 0, 0, 0]], columns=["video_id", "label", "start", "end"])
    data = data.reset_index(drop=True)
    data = data.sort_values(by=["video_id", "label"])
    for i in data["video_id"].unique():
        data_video = data[data["video_id"]==i]
        list_label = data_video["label"].unique()
        vid_all = pd.DataFrame([[0, 0, 0, 0]], columns=["video_id", "label", "start", "end"])
        for label in list_label:

            data_video_label = data_video[data_video["label"]== label]
            data_video_label = data_video_label.reset_index()
            data_video_label = data_video_label.sort_values(by=["start"])

            for j in range(len(data_video_label)-1):

                if data_video_label.loc[j+1, "start"] - data_video_label.loc[j, "end"] <= merge_threshold:
                    data_video_label.loc[j+1, "start"] = data_video_label.loc[j, "start"]
                    data_video_label.loc[j, "end"] = 0
                    data_video_label.loc[j, "start"] = 0

            vid_all = vid_all.append(data_video_label)
        vid_all = vid_all[vid_all["end"]!=0]
        print("vid", i)
        vid_all = process_overlap(vid_all)
        # print("p2")
        # print(vid_all.sort_values(by=["start"]).reset_index(drop=True))

        # vid_all = process_overlap(vid_all)
        df_total = df_total.append(vid_all)
    df_total = df_total[df_total["end"]!=0]
    # 对于除捡东西以外其它类别， 小于等于 2s 去除

    df_short =  df_total[df_total["label"].isin([9, 10])]
    df_long = df_total[~df_total["label"].isin([9, 10])]
    df_long  = df_long[(df_long["end"] - df_long["start"] > 2)]


    df_total = pd.concat([df_short, df_long], join="inner")

    df_total = df_total.drop(columns=['index'])
    df_total = df_total.sort_values(by=["video_id", "start"])
    # df_total.to_csv("./output/A1_val_sample.csv", index=False, columns=["video_id", "label", "start", "end"])
    return df_total


def general_submission(data):
    # data = pd.read_csv(filename, sep=" ", header=None)
    data_filtered = data[data["label"] != 0]
    data_filtered["start"] = data["start"].map(lambda x: int(float(x)))
    data_filtered["end"] = data["end"].map(lambda x: int(float(x)))
    data_filtered = data_filtered.sort_values(by=["video_id","label"])
    # data_filtered.to_csv(r'./output/AIC_1004_ensemble_3view_1s.txt', header=None, index=None, sep=' ', mode='w')
    results = merge_and_remove(data_filtered, merge_threshold=8)
    return results


def topk_by_partition(input, k, axis=None, ascending=False):
    if not ascending:
        input *= -1
    ind = np.argpartition(input, k, axis=axis)
    ind = np.take(ind, np.arange(k), axis=axis) # k non-sorted indices
    input = np.take_along_axis(input, ind, axis=axis) # k non-sorted values

    # sort within k elements
    ind_part = np.argsort(input, axis=axis)
    ind = np.take_along_axis(ind, ind_part, axis=axis)
    if not ascending:
        input *= -1
    val = np.take_along_axis(input, ind_part, axis=axis) 
    return ind, val


def get_classification(sequence_class_prob):
    classify=[[x,y] for x,y in zip(np.argmax(sequence_class_prob, axis=1),np.max(sequence_class_prob, axis=1))]
    labels_index = np.argmax(sequence_class_prob, axis=1) #returns list of position of max value in each list.
    probs= np.max(sequence_class_prob, axis=1)  # return list of max value in each  list.
    return labels_index, probs
    # labels_index = []
    # probs = []
    # for i in range(sequence_class_prob.shape[0]):
    #     pred_top2, prob_top2 = topk_by_partition(sequence_class_prob[i].copy(), 2, axis=-1)
    #     pred_top2 = pred_top2.flatten()
    #     prob_top2 = prob_top2.flatten()
    #     if pred_top2[0] == 0 and prob_top2[1] > 0.8:
    #         pred = pred_top2[1]
    #         labels_index.append(pred)
    #         probs.append(prob_top2[1])
    #     else:
    #         pred = pred_top2[0]
    #         labels_index.append(pred)
    #         probs.append(prob_top2[0]) 
    # return np.array(labels_index), np.array(probs)

def activity_localization(prob_sq, action_threshold):
    """
    利用阈值对时序得分曲线二值化，确定最有可能含有动作的时间区间
    """
    action_idx, action_probs = get_classification(prob_sq)
    threshold = np.mean(action_probs)
    action_tag = np.zeros(action_idx.shape)
    # action_tag[action_probs >= threshold] = 1
    action_tag[action_probs >= action_threshold] = 1
    # print('action_tag', action_tag)
    activities_idx = []
    startings = []
    endings = []

    for i in range(len(action_tag)):
        if action_tag[i] ==1:
            activities_idx.append(action_idx[i])
            start = i
            end = i+1
            startings.append(start)
            endings.append(end)
    return activities_idx, startings, endings

def smoothing(x, k=3):
    ''' Applies a mean filter to an input sequence. The k value specifies the window
    size. window size = 2*k
    '''
    l = len(x)
    s = np.arange(-k, l - k)
    e = np.arange(k, l + k)
    s[s < 0] = 0
    e[e >= l] = l - 1
    y = np.zeros(x.shape)
    for i in range(l):
        y[i] = np.mean(x[s[i]:e[i]], axis=0)
    return y

def gauss_smoothing(x, k=3):
    ''' Applies a mean filter to an input sequence. The k value specifies the window
    size. window size = 2*k
    '''
    l = len(x)

    s = np.arange(-k, l - k)
    e = np.arange(k, l + k)
    s[s < 0] = 0
    e[e >= l] = l - 1

    f = np.zeros(k*2, dtype=np.float32)
    total = 0
    for i in range(-k, k):
        f[i+k] = np.exp(-(i/2)**2)
        total += f[i+k]
    f = f / total
    f = f.reshape(-1, 1)

    y = np.zeros(x.shape)
    for i in range(l):
        if e[i] - s[i] < 2*k:
            y[i] = np.mean(x[s[i]:e[i]], axis=0)
        else:
            y[i] = np.sum(x[s[i]:e[i]]*f, axis=0)
    return y

def compute_os_score(ground_truth, prediction):
    ground_truth_gbvn = ground_truth.groupby('video_id')
    label = ground_truth["label"].unique()
    scores = []
    for idx, this_pred in prediction.iterrows():
        video_id = this_pred['video_id']
        try:
            this_gt = ground_truth_gbvn.get_group(int(video_id))
            this_gt = this_gt.reset_index()
            tiou_arr = util_eval.segment_iou(this_pred[["start", "end"]].values, this_gt[["start", "end"]].values)
            scores += [item for item in tiou_arr if item > 0]
        except:
            print("Video {} gt has no {} action".format(video_id, label))
    return scores



# with open("classification_probs/dash_prob_8x8_ema.pkl", "rb") as fp:
#     dash_prob_8x8 = pickle.load(fp)

# with open("classification_probs/rear_prob_8x8_ema.pkl", "rb") as fp:
#     rear_prob_8x8 = pickle.load(fp)


# with open("classification_probs/right_prob_8x8_ema.pkl", "rb") as fp:
#     right_prob_8x8 = pickle.load(fp)



# with open("right_vmae_16x4_cor.pkl", "rb") as fp:
#     right_prob_16x4_2 = pickle.load(fp)

with open("pickles/right_vmae_16x4_cor.pkl", "rb") as fp:
    right_prob_16x4 = pickle.load(fp)


with open("pickles/dash_vmae_16x4_cor.pkl", "rb") as fp:
    dash_prob_16x4 = pickle.load(fp)

with open("pickles/rear_vmae_16x4_cor.pkl", "rb") as fp:
    rear_prob_16x4 = pickle.load(fp)

# with open("pickles/right_vmae_16x4_cor_s15.pkl", "rb") as fp:
#     right_prob_16x4 = pickle.load(fp)


# with open("pickles/dash_vmae_16x4_cor_s15.pkl", "rb") as fp:
#     dash_prob_16x4 = pickle.load(fp)

# with open("pickles/rear_vmae_16x4_cor_s15.pkl", "rb") as fp:
#     rear_prob_16x4 = pickle.load(fp)



with open("pickles/right_vmae_16x4_crop.pkl", "rb") as fp:
    right_prob_16x4_crop = pickle.load(fp)

with open("pickles/dash_vmae_16x4_crop.pkl", "rb") as fp:
    dash_prob_16x4_crop = pickle.load(fp)
with open("pickles/rear_vmae_16x4_crop.pkl", "rb") as fp:
    rear_prob_16x4_crop = pickle.load(fp)


with open("pickles/A1_newgt/A1_right_vmae_16x4_crop.pkl", "rb") as fp:
    right_prob_16x4_crop = pickle.load(fp)

with open("pickles/A1_newgt/A1_dash_vmae_16x4_crop.pkl", "rb") as fp:
    dash_prob_16x4_crop = pickle.load(fp)

with open("pickles/A1_newgt/A1_rear_vmae_16x4_crop.pkl", "rb") as fp:
    rear_prob_16x4_crop = pickle.load(fp)




class_names = [
"Drinking",
"Phone Call（Right）",
"Phone Call（Left）",
"Eating",
"Text(Right)",
"Text(Left)",
"Reaching behind",
"Adjust Control Pane",
"Pick up from floor(Driver)",
"Pick up from floor(Passenger)",
"Talk to passenger at the right",
"Talk to passenger at the backseat",
"yawning",
"Hand on head",
"Singing or dance with music"
]
class_map = dict([(str(i+1), class_names[i]) for i in range(len(class_names))])



def main(alpha, beta, sigma):
    mapping = {
        "Right_side_window_user_id_28557_NoAudio_5": 1,
        "Right_side_window_user_id_28557_NoAudio_7": 2,
        "Right_side_window_user_id_31903_NoAudio_5": 3,
        "Right_side_window_user_id_31903_NoAudio_7": 4, 
        "Right_side_window_user_id_59581_NoAudio_5": 5, 
        "Right_side_window_user_id_59581_NoAudio_7": 6,
        "Right_side_window_user_id_85870_NoAudio_5": 7,
        "Right_side_window_user_id_85870_NoAudio_7": 8,
        "Right_side_window_user_id_83323_NoAudio_3": 9,
        "Right_side_window_user_id_83323_NoAudio_5": 10,
        "Dashboard_user_id_28557_NoAudio_5": 1,
        "Dashboard_user_id_28557_NoAudio_7": 2,
        "Dashboard_user_id_31903_NoAudio_5": 3,
        "Dashboard_user_id_31903_NoAudio_7": 4, 
        "Dashboard_user_id_59581_NoAudio_5": 5, 
        "Dashboard_user_id_59581_NoAudio_7": 6,
        "Dashboard_user_id_85870_NoAudio_5": 7,
        "Dashboard_user_id_85870_NoAudio_7": 8,
        "Dashboard_user_id_83323_NoAudio_3": 9,
        "Dashboard_user_id_83323_NoAudio_5": 10,
        "Rear_view_user_id_28557_NoAudio_5": 1,
        "Rear_view_user_id_28557_NoAudio_7": 2,
        "Rear_view_user_id_31903_NoAudio_5": 3,
        "Rear_view_user_id_31903_NoAudio_7": 4, 
        "Rear_view_user_id_59581_NoAudio_5": 5, 
        "Rear_view_user_id_59581_NoAudio_7": 6,
        "Rear_view_user_id_85870_NoAudio_5": 7,
        "Rear_view_user_id_85870_NoAudio_7": 8,
        "Rear_view_user_id_83323_NoAudio_3": 9,
        "Rear_view_user_id_83323_NoAudio_5": 10

    }
    classification = []
    localization = []
    for right_vid in right_prob_16x4.keys():
        dash_vid = "Dashboard_"+re.search("user_id_[0-9]{5}_NoAudio_[0-9]", right_vid)[0]
        rear_vid = "Rear_view_"+re.search("user_id_[0-9]{5}_NoAudio_[0-9]", right_vid)[0]
        all_dash_probs = np.array(dash_prob_16x4[dash_vid])
        all_right_probs = np.array(right_prob_16x4[right_vid])
        all_rear_probs = np.array(rear_prob_16x4[rear_vid])

        # all_dash_probs =[]
        # for i in range(0, len(all_dash_probs_temp), 2):
        #     all_dash_probs.append(np.array(all_dash_probs_temp[i])*0.5+np.array(all_dash_probs_temp[i+1])*0.5)

        # all_right_probs =[]
        # for i in range(0, len(all_right_probs_temp), 2):
        #     all_right_probs.append(np.array(all_right_probs_temp[i])*0.5+np.array(all_right_probs_temp[i+1])*0.5)
            

        # all_rear_probs =[]
        # for i in range(0, len(all_rear_probs_temp), 2):
        #     all_rear_probs.append(np.array(all_rear_probs_temp[i])*0.5+np.array(all_rear_probs_temp[i+1])*0.5)
            


        all_dash_probs_crop = np.array(dash_prob_16x4_crop[dash_vid])
        all_right_probs_crop = np.array(right_prob_16x4_crop[right_vid])
        all_rear_probs_crop = np.array(rear_prob_16x4_crop[rear_vid])
        prob_ensemble = []
        for t in range(min(len(all_rear_probs), len(all_dash_probs), len(all_right_probs), len(all_dash_probs_crop))):
            # prob1 = (all_dash_probs[t] + dash_prob_16x4[int(vid)][t])/2 if t<len(dash_prob_16x4[int(vid)]) else all_dash_probs[t]
            # prob2 = (all_rear_probs[t] + rear_prob_16x4[int(vid)][t])/2 if t<len(rear_prob_16x4[int(vid)]) else all_rear_probs[t]

            # prob_avg = np.array((all_dash_probs[t]+all_dash_probs_crop[t])/2) * alpha + np.array((all_right_probs[t]+all_right_probs_crop[t])/2) * beta + np.array((all_rear_probs[t]+all_rear_probs_crop[t])/2) * sigma
           
            prob_avg = np.array(all_dash_probs_crop[t]) * alpha + np.array(all_right_probs_crop[t]) * beta + np.array(all_rear_probs_crop[t]) * sigma
           
            # prob_avg = np.array(all_right_probs_crop[t])
            # prob_avg = np.array(all_dash_probs_crop[t]) * alpha + np.array(all_right_probs_crop[t]) * beta + np.array(all_rear_probs_crop[t]) * sigma
            # prob_avg = np.array(all_dash_probs[t]) * alpha + np.array(all_right_probs[t]) * beta + np.array(all_rear_probs[t]) * sigma
            # prob_avg = np.amax(np.stack([all_dash_probs[t], all_right_probs[t], all_rear_probs[t]], axis=0), axis=0)
            # prob_avg = all_rear_probs[t]
            # print(prob1.shape, dash_w.shape, prob2.shape, rear_w.shape, prob3.shape, right_w.shape)
            # prob_avg = all_dash_probs[t] * dash_w + all_right_probs[t] * right_w 
            # prob_avg = (all_right_probs[t]+ all_rear_probs[t] + all_dash_probs[t])/3
            prob_ensemble.append(prob_avg)
            
        prob_seq = np.array(prob_ensemble)
        prob_seq = np.squeeze(prob_seq)
        vid = mapping[dash_vid] 
        #TODO: visualize prob curve
        # plot_probs(prob_seq, vid)
        activities_idx, startings, endings = activity_localization(prob_seq, action_threshold=0.1)
        for label, s, e in zip(activities_idx, startings, endings):
            start = s * 30/30.
            end = e * 30/30
            classification.append([int(vid), label, start, end])

        prob_seq_smooth = smoothing(prob_seq, k=1) 
        # plot_probs(prob_seq_smooth, str(vid)+"_smooth")
        activities_idx, startings, endings = activity_localization(prob_seq_smooth, action_threshold=0.1)
        for label, s, e in zip(activities_idx, startings, endings):
            start = s * 30/30.
            end = e * 30/30.
            localization.append([int(vid), label, start, end])

    classification = pd.DataFrame(classification, columns =["video_id", "label", "start", "end"])
    rough_loc = pd.DataFrame(localization, columns =["video_id", "label", "start", "end"])
    prediction = general_submission(rough_loc)
    
    classification.to_csv("cls.csv", columns =["video_id", "label", "start", "end"], index=False)
    rough_loc.to_csv("rough_loc.csv", columns =["video_id", "label", "start", "end"], index=False)
    prediction.to_csv("pred.csv", columns =["video_id", "label", "start", "end"], index=False)
    # load pred file
    CLASS_NUM = 15
    data_root = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/qianyinlong/challenge/AI-City-2023/Naturalistic-Driving-Action-Recognition/Data/Ori-data/A1"
    gt = pd.read_csv(os.path.join(data_root, "val_segments_gt.csv"))

    M, N = len(gt), len(prediction)
    gt_by_label = gt.groupby("label")
    pred_by_label = prediction.groupby("label")

    scores = []
    for label in range(1, CLASS_NUM+1):
        try:
            ground_truth_class = gt_by_label.get_group(label).reset_index(drop=True)
            prediction_class = pred_by_label.get_group(label).reset_index(drop=True)   
            scores += compute_os_score(ground_truth_class, prediction_class)
        except:
            continue
    print("Total Action:", M)
    print("True Positive:", len(scores))
    print("False Positive:", N-len(scores))
    print("False Negtive:", M-len(scores))
    print("score", sum(scores) / (M+N-len(scores)))
    print("PRECISON AVG", sum(scores)/len(scores))
    print("Recall AVG", sum(scores)/M)
    print("-"*50 + "visualize"+"-"*50)

    # fig = plt.figure(figsize=(24,10))
    # unique_video_id = np.array(np.unique(gt.video_id.values))

    # for idx in range(len(unique_video_id)):
    #     video_id = int(unique_video_id[idx])
    #     ax = fig.add_subplot(len(unique_video_id), 1, idx+1)

    #     ax.invert_yaxis()
    #     # ax.xaxis.set_visible(False)
    #     ax.set_xlim(0, 540)


    #     pred = classification[classification["video_id"]==video_id]
    #     pred_starts = pred['start']
    #     pred_widths = pred['end'] - pred['start'] + 1
    #     pred_labels = pred['label']
    #     for label, start, width in zip(pred_labels, pred_starts, pred_widths):
    #         if label == 0:
    #             continue
    #         rects = ax.barh("Cls-{}".format(video_id), width, left=start, height=0.4, color=cmap[int(label)])


    #     pred = rough_loc[rough_loc["video_id"]==video_id]
    #     pred_starts = pred['start']
    #     pred_widths = pred['end'] - pred['start'] + 1
    #     pred_labels = pred['label']
    #     for label, start, width in zip(pred_labels, pred_starts, pred_widths):
    #         if label == 0:
    #             continue
    #         rects = ax.barh("Rough-{}".format(video_id), width, left=start, height=0.4, color=cmap[int(label)])


    #     pred = prediction[prediction["video_id"]==video_id]
    #     pred_starts = pred['start']
    #     pred_widths = pred['end'] - pred['start'] + 1
    #     pred_labels = pred['label']
    #     for label, start, width in zip(pred_labels, pred_starts, pred_widths):
    #         if label == 0:
    #             continue
    #         rects = ax.barh("Pred-{}".format(video_id), width, left=start, height=0.4, color=cmap[int(label)])

    #     gt_vid = gt[gt["video_id"]==video_id]
    #     gt_starts = gt_vid['start']
    #     gt_widths = gt_vid['end'] - gt_vid['start'] + 1
    #     gt_labels = gt_vid['label']

    #     for label, start, width in zip(gt_labels, gt_starts, gt_widths):
    #         if label == 0:
    #             continue
    #         rects = ax.barh("GroundTruth-{}".format(video_id), width, left=start, height=0.4, color=cmap[int(label)])


    # for label in range(1, 16):
    #     width = 1
    #     rects = ax.barh("Label", width, left=0, height=0.2, color=cmap[int(label)], label=class_map[str(label)])
    # plt.xticks(np.linspace(0, 540, 55).astype(np.int32))
    # plt.legend(ncols=1, bbox_to_anchor=(0, 1),
    #         loc='right', fontsize='small')
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.savefig("vis_a{}b{}s{}.png".format(alpha, beta, sigma),bbox_inches="tight")

for i in range(0, 10+1):
    for j in range(0, 10-i+1):
        alpha = i / 10
        beta = j / 10
        sigma = 1 - alpha - beta
        print("alpha {}, beta {}, sigma {}".format(alpha, beta, sigma))
        main(alpha, beta, sigma)
# alpha = 1.0
# beta = 0.0
# sigma = 0.0
# main(alpha, beta, sigma)