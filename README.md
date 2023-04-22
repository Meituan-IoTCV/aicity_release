# Implementation of Multi view Action Recognition for Distracted Driver Behavior Localization

This repository is the official implementation of [Multi view Action Recognition for Distracted Driver Behavior Localization](pdf/19.pdf).  This paper presents our approach for Track 3 (Natural- istic Driving Action Recognition) of the 2023 AI City Challenge, where the objective is to classify distracting driving activities in each untrimmed naturalistic driving video and localize the accurate temporal boundaries of them. We  rank the 1st on the Test-A2 of the challenge track.
![pipeline](figs/pipeline.png)
## Requirements
Please follow the installation instructions in [VideoMAE](https://github.com/MCG-NJU/VideoMAE). Also, you can simply run the following command:
```
conda env create -f environment.yml
conda activate pt1.9.0cu11.1_official
```

##  Data Preprocessing
**1. Data download**
Download the dataset from the offical website. Next, put "A1", "A2" split into folders "data/A1", "data/A2" respectively.   


**2. Action segments extraction**
Given the annotation file, we extract 16 distracted action classes from the video, which will be used as the recognition model training data. Run the following command:

```
python preprocess/extract_clips.py
```
and the video segments will be saved in folder "data/A1_clip".

**3. Generate K-Fold "train/val" split.**

```
python preprocess/split_k_fold.py
```




## Training Recognition Model
Our model is initialized with pretrained VideoMAE with  Kinetic-710, you can get the pretrained weights from [VideoMAE-Kinetic-710/ViT-L](https://drive.google.com/file/d/1jX1CiqxSkCfc94y8FRW1YGHy-GNvHCuD/view?usp=sharing) 
Modified the "MODEL_PATH" in script "scrpts/cls/train_cls.sh" to your own path.
To train the model (s) in the paper, run this command:

```train
bash sripts/cls/train_cls.sh
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Inference A2 video

To evaluate my model on ImageNet, run:

```inference 
bash scripts/cls/inference_cls.sh 
```


## Postprocessing 

After inferencing the "test" set videos, we get the classification probablity sequence for each video. To get the final location results, we need to perform postprocessing on  
```
python run_submission.py
```


## Customed Data 
 

## Model Zoo
We release our trained model weights organized with "camera view" and "fold k". You can download to reproduce the results we report in the paper.

| Camera View       | Fold | Model Results |
| ------------------ |---------------- | -------------- |
| Dashboard   |     0         |       [dash_0.pth](https://huggingface.co/idealweiwei/aicity_release/tree/main)    |
| Dashboard   |     1         |       [dash_1.pth](https://huggingface.co/idealweiwei/aicity_release/tree/main)    |
| Dashboard   |     2         |       [dash_2.pth](https://huggingface.co/idealweiwei/aicity_release/tree/main)    |
| Dashboard   |     3         |       [dash_3.pth](https://huggingface.co/idealweiwei/aicity_release/tree/main)    |
| Dashboard   |     4         |       [dash_4.pth](https://huggingface.co/idealweiwei/aicity_release/tree/main)    |
| Rightside   |     0         |       [rightside_0.pth](https://huggingface.co/idealweiwei/aicity_release/tree/main)    |
| Rightside   |     1         |       [rightside_1.pth](https://huggingface.co/idealweiwei/aicity_release/tree/main)    |
| Rightside   |     2         |       [rightside_2.pth](https://huggingface.co/idealweiwei/aicity_release/tree/main)    |
| Rightside   |     3         |       [rightside_3.pth](https://huggingface.co/idealweiwei/aicity_release/tree/main)    |
| Rightside   |     4         |       [rightside_4.pth](https://huggingface.co/idealweiwei/aicity_release/tree/main)    |
| Rear View   |     0         |       [rearview_0.pth](https://huggingface.co/idealweiwei/aicity_release/tree/main)    |
| Rear View   |     1         |       [rearview_1.pth](https://huggingface.co/idealweiwei/aicity_release/tree/main)    |
| Rear View   |     2         |       [rearview_2.pth](https://huggingface.co/idealweiwei/aicity_release/tree/main)    |
| Rear View   |     3         |       [rearview_3.pth](https://huggingface.co/idealweiwei/aicity_release/tree/main)    |
| Rear View   |     4         |       [rearview_4.pth](https://huggingface.co/idealweiwei/aicity_release/tree/main)    |


## Contact
For further discussion, you are welcomed to send an e-mail to the following email address. 
zhouwei85@meituan.com
qianyinlong@meituan.com