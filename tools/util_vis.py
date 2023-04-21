
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

_CLASS_NAME = [
    "Drinking",         
    "Phone Call（Right）",
    "Phone Call（Left）",
    "Eating",       # dash
    "Text(Right)",   
    "Text(Left)",
    "Reaching behind",
    "Adjust Control Pane",
    "Pick up from floor(Driver)",
    "Pick up from floor(Passenger)",
    "Talk to passenger at the right",
    "Talk to passenger at the backseat",
    "yawning",      # dash 
    "Hand on head", # dashg
    "Singing or dance with music" # dash
]
_CLASS_MAP = dict([(str(i+1), _CLASS_NAME[i]) for i in range(len(_CLASS_NAME))])


_LABEL_COLOURS = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]

_COLOR_MAP = np.asarray(_LABEL_COLOURS) / 255.0


def plot_probs(probs, video_id):
    class_names = [
    "Normal",#0
    "Drinking",#1
    "Phone Call（Right）",#2
    "Phone Call（Left）",#3
    "Eating",#4
    "Text(Right)",#5
    "Text(Left)",#6
    "Reaching behind",#7
    "Adjust Control Pane",#8
    "Pick up from floor(Driver)", #9
    "Pick up from floor(Passenger)",#10
    "Talk to passenger at the right",#11
    "Talk to passenger at the backseat",#12
    "yawning",#13
    "Hand on head",#14
    "Singing or dance with music"#15
    ]
    NUM_CLASSES = probs.shape[1]
    seq_len = probs.shape[0]
    fig = plt.figure(figsize=(24, 24))
    for i in range(NUM_CLASSES):
        ax = fig.add_subplot(NUM_CLASSES, 1, i+1)
        ax.plot(list(range(seq_len)), probs[:, i])
        ax.set_xlabel("{}".format(class_names[i]))
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("{}_probs.png".format(video_id), bbox_inches="tight")


def vis_loc_result(clip_level_classification, prediction, gt):
    fig = plt.figure(figsize=(24,10))
    unique_video_id = np.array(np.unique(prediction.video_id.values))

    for idx in range(len(unique_video_id)):
        video_id = int(unique_video_id[idx])
        ax = fig.add_subplot(len(unique_video_id), 1, idx+1)

        ax.invert_yaxis()
        # ax.xaxis.set_visible(False)
        ax.set_xlim(0, 540)


        clc = clip_level_classification[clip_level_classification["video_id"]==video_id]
        clc_starts = clc['start']
        clc_widths = clc['end'] - clc['start'] + 1
        clc_labels = clc['label']
        for label, start, width in zip(clc_labels, clc_starts, clc_widths):
            if label == 0:
                continue
            rects = ax.barh("CLC-{}".format(video_id), width, left=start, height=0.4, color=_COLOR_MAP[int(label)])


        pred = prediction[prediction["video_id"]==video_id]
        pred_starts = pred['start']
        pred_widths = pred['end'] - pred['start'] + 1
        pred_labels = pred['label']
        for label, start, width in zip(pred_labels, pred_starts, pred_widths):
            if label == 0:
                continue
            rects = ax.barh("Pred-{}".format(video_id), width, left=start, height=0.4, color=_COLOR_MAP[int(label)])


        gt_vid = gt[gt["video_id"]==video_id]
        gt_starts = gt_vid['start']
        gt_widths = gt_vid['end'] - gt_vid['start'] + 1
        gt_labels = gt_vid['label']

        for label, start, width in zip(gt_labels, gt_starts, gt_widths):
            if label == 0:
                continue
            rects = ax.barh("GroundTruth-{}".format(video_id), width, left=start, height=0.4, color=_COLOR_MAP[int(label)])

    for label in range(1, 16):
        width = 1
        rects = ax.barh("Label", width, left=0, height=0.2, color=_COLOR_MAP[int(label)], label=_CLASS_MAP[str(label)])
    plt.xticks(np.linspace(0, 540, 55).astype(np.int32))
    plt.legend(ncols=1, bbox_to_anchor=(0, 1),
            loc='right', fontsize='small')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("vis_A2.png",bbox_inches="tight")
