
# Set the path to save checkpoints
OUTPUT_DIR='../../model_zoo/k400_videomae_pretrain_large_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/eval_lr_2e-3_repeated_aug_epoch_35+0304_lion_dash_betas_8x8'
# path to Kinetics set (train.csv/val.csv/test.csv)
DATA_PATH='/home/hadoop-vacv/hadoop-vacv/qianyinlong/challenge/AI-City-2023/Naturalistic-Driving-Action-Recognition/Data/Ori-data/A1_clip/videomae/'
# # new gt, cropped, fold1
MODEL_PATH="../../model_zoo/k400_intervideo_pretrain_large_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/K400_init_right_view_baseline_newgt_fold2/checkpoint-34.pth"
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=5 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay 0.05 \
    --epochs 35 \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --lr 2e-3 \
    --view "right" \
    --clip_stride 30 \
    --fold 2 \
    --crop \

MODEL_PATH="../../model_zoo/k400_intervideo_pretrain_large_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/K400_init_dash_view_baseline_newgt_fold2/checkpoint-34.pth"
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=5 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay 0.05 \
    --epochs 35 \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --lr 2e-3 \
    --view "dash" \
    --clip_stride 30 \
    --fold 2 \
    --crop \



MODEL_PATH="../../model_zoo/k400_intervideo_pretrain_large_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/K400_init_rear_view_baseline_newgt_fold2/checkpoint-34.pth"
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=5 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay 0.05 \
    --epochs 35 \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --lr 2e-3 \
    --view "rear" \
    --clip_stride 30 \
    --fold 2 \
    --crop \
