# Set the path to save checkpoints
# OUTPUT_DIR='YOUR_PATH/k400_videomae_pretrain_large_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/eval_lr_2e-3_repeated_aug_epoch_35'
# # path to Kinetics set (train.csv/val.csv/test.csv)
# DATA_PATH='YOUR_PATH/list_kinetics-400'
# # path to pretrain model
# MODEL_PATH='YOUR_PATH/k400_videomae_pretrain_large_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/checkpoint-1599.pth'

# We add repeated_aug (--num_sample = 2) on Kinetics-400 here, 
# which could better performance while need more time for fine-tuning

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
#     --master_port 12320 --nnodes=8  --node_rank=$1 --master_addr=$2 \
#     run_class_finetuning.py \
#     --model vit_large_patch16_224 \
#     --data_set Kinetics-400 \
#     --nb_classes 400 \
#     --data_path ${DATA_PATH} \
#     --finetune ${MODEL_PATH} \
#     --log_dir ${OUTPUT_DIR} \
#     --output_dir ${OUTPUT_DIR} \
#     --batch_size 2 \
#     --num_sample 2 \
#     --input_size 224 \
#     --short_side_size 224 \
#     --save_ckpt_freq 10 \
#     --num_frames 16 \
#     --sampling_rate 4 \
#     --opt adamw \
#     --lr 2e-3 \
#     --opt_betas 0.9 0.999 \
#     --weight_decay 0.05 \
#     --epochs 35 \
#     --dist_eval \
#     --test_num_segment 5 \
#     --test_num_crop 3 \
#     --enable_deepspeed 



# Set the path to save checkpoints
# path to Kinetics set (train.csv/val.csv/test.csv)
DATA_PATH='/home/hadoop-vacv/hadoop-vacv/qianyinlong/challenge/AI-City-2023/Naturalistic-Driving-Action-Recognition/Data/Ori-data/A1_clip_v2/videomae/'
# path to pretrain model
# MODEL_PATH='/home/hadoop-vacv/hadoop-vacv/zhouwei/code/models/ViTL-Kinetics.pth'
MODEL_PATH='/home/hadoop-vacv/hadoop-vacv/zhouwei/code/models/vit_l_hybrid_pt_800e_k700_ft.pth'

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
# OUTPUT_DIR='../../model_zoo/k400_intervideo_pretrain_large_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/K400_init_right_view_baseline_newgt_fold4'
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node=5 \
#     --master_port 12320 --nnodes=1  --node_rank=0 --master_addr=localhost \
#     run_class_finetuning.py \
#     --model vit_large_patch16_224 \
#     --data_set Kinetics-400 \
#     --nb_classes 16 \
#     --data_path ${DATA_PATH} \
#     --finetune ${MODEL_PATH} \
#     --log_dir ${OUTPUT_DIR} \
#     --output_dir ${OUTPUT_DIR} \
#     --batch_size 2 \
#     --num_sample 1 \
#     --input_size 224 \
#     --short_side_size 224\
#     --save_ckpt_freq 5 \
#     --num_frames 16 \
#     --sampling_rate 4 \
#     --opt lion \
#     --opt_betas 0.9 0.999 \
#     --warmup_epochs 5 \
#     --epochs 35 \
#     --test_num_segment 10 \
#     --test_num_crop 1 \
#     --lr 2e-3 \
#     --view "right_4" \
#     --layer_decay 0.75 \
#     --dist_eval \
#     --fc_drop_rate 0.5 \


# OUTPUT_DIR='../../model_zoo/k400_intervideo_pretrain_large_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/K400_init_dash_view_baseline_newgt_fold4'
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node=5 \
#     --master_port 12320 --nnodes=1  --node_rank=0 --master_addr=localhost \
#     run_class_finetuning.py \
#     --model vit_large_patch16_224 \
#     --data_set Kinetics-400 \
#     --nb_classes 16 \
#     --data_path ${DATA_PATH} \
#     --finetune ${MODEL_PATH} \
#     --log_dir ${OUTPUT_DIR} \
#     --output_dir ${OUTPUT_DIR} \
#     --batch_size 2 \
#     --num_sample 1 \
#     --input_size 224 \
#     --short_side_size 224\
#     --save_ckpt_freq 5 \
#     --num_frames 16 \
#     --sampling_rate 4 \
#     --opt lion \
#     --opt_betas 0.9 0.999 \
#     --warmup_epochs 5 \
#     --epochs 35 \
#     --test_num_segment 10 \
#     --test_num_crop 1 \
#     --lr 2e-3 \
#     --view "dash_4" \
#     --layer_decay 0.75 \
#     --dist_eval \
#     --fc_drop_rate 0.5 \


# OUTPUT_DIR='../../model_zoo/k400_intervideo_pretrain_large_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/K400_init_rear_view_baseline_newgt_fold4'
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node=5 \
#     --master_port 12320 --nnodes=1  --node_rank=0 --master_addr=localhost \
#     run_class_finetuning.py \
#     --model vit_large_patch16_224 \
#     --data_set Kinetics-400 \
#     --nb_classes 16 \
#     --data_path ${DATA_PATH} \
#     --finetune ${MODEL_PATH} \
#     --log_dir ${OUTPUT_DIR} \
#     --output_dir ${OUTPUT_DIR} \
#     --batch_size 2 \
#     --num_sample 1 \
#     --input_size 224 \
#     --short_side_size 224\
#     --save_ckpt_freq 5 \
#     --num_frames 16 \
#     --sampling_rate 4 \
#     --opt lion \
#     --opt_betas 0.9 0.999 \
#     --warmup_epochs 5 \
#     --epochs 35 \
#     --test_num_segment 10 \
#     --test_num_crop 1 \
#     --lr 2e-3 \
#     --view "rear_4" \
#     --layer_decay 0.75 \
#     --dist_eval \
#     --fc_drop_rate 0.5 \



# OUTPUT_DIR='../../model_zoo/k400_intervideo_pretrain_large_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/K400_init_right_view_baseline_newgt_fold3'
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node=5 \
#     --master_port 12320 --nnodes=1  --node_rank=0 --master_addr=localhost \
#     run_class_finetuning.py \
#     --model vit_large_patch16_224 \
#     --data_set Kinetics-400 \
#     --nb_classes 16 \
#     --data_path ${DATA_PATH} \
#     --finetune ${MODEL_PATH} \
#     --log_dir ${OUTPUT_DIR} \
#     --output_dir ${OUTPUT_DIR} \
#     --batch_size 2 \
#     --num_sample 1 \
#     --input_size 224 \
#     --short_side_size 224\
#     --save_ckpt_freq 5 \
#     --num_frames 16 \
#     --sampling_rate 4 \
#     --opt lion \
#     --opt_betas 0.9 0.999 \
#     --warmup_epochs 5 \
#     --epochs 35 \
#     --test_num_segment 10 \
#     --test_num_crop 1 \
#     --lr 2e-3 \
#     --view "right_3" \
#     --layer_decay 0.75 \
#     --dist_eval \
#     --fc_drop_rate 0.5 \


# OUTPUT_DIR='../../model_zoo/k400_intervideo_pretrain_large_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/K400_init_dash_view_baseline_newgt_fold3'
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node=5 \
#     --master_port 12320 --nnodes=1  --node_rank=0 --master_addr=localhost \
#     run_class_finetuning.py \
#     --model vit_large_patch16_224 \
#     --data_set Kinetics-400 \
#     --nb_classes 16 \
#     --data_path ${DATA_PATH} \
#     --finetune ${MODEL_PATH} \
#     --log_dir ${OUTPUT_DIR} \
#     --output_dir ${OUTPUT_DIR} \
#     --batch_size 2 \
#     --num_sample 1 \
#     --input_size 224 \
#     --short_side_size 224\
#     --save_ckpt_freq 5 \
#     --num_frames 16 \
#     --sampling_rate 4 \
#     --opt lion \
#     --opt_betas 0.9 0.999 \
#     --warmup_epochs 5 \
#     --epochs 35 \
#     --test_num_segment 10 \
#     --test_num_crop 1 \
#     --lr 2e-3 \
#     --view "dash_3" \
#     --layer_decay 0.75 \
#     --dist_eval \
#     --fc_drop_rate 0.5 \


# OUTPUT_DIR='../../model_zoo/k400_intervideo_pretrain_large_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/K400_init_rear_view_baseline_newgt_fold3'
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node=5 \
#     --master_port 12320 --nnodes=1  --node_rank=0 --master_addr=localhost \
#     run_class_finetuning.py \
#     --model vit_large_patch16_224 \
#     --data_set Kinetics-400 \
#     --nb_classes 16 \
#     --data_path ${DATA_PATH} \
#     --finetune ${MODEL_PATH} \
#     --log_dir ${OUTPUT_DIR} \
#     --output_dir ${OUTPUT_DIR} \
#     --batch_size 2 \
#     --num_sample 1 \
#     --input_size 224 \
#     --short_side_size 224\
#     --save_ckpt_freq 5 \
#     --num_frames 16 \
#     --sampling_rate 4 \
#     --opt lion \
#     --opt_betas 0.9 0.999 \
#     --warmup_epochs 5 \
#     --epochs 35 \
#     --test_num_segment 10 \
#     --test_num_crop 1 \
#     --lr 2e-3 \
#     --view "rear_3" \
#     --layer_decay 0.75 \
#     --dist_eval \
#     --fc_drop_rate 0.5 \




# OUTPUT_DIR='../../model_zoo/k400_intervideo_pretrain_large_patch16_224_frame_16x8_tube_mask_ratio_0.9_e1600/K400_init_dash_view_baseline_newgt_fold0'
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node=5 \
#     --master_port 12320 --nnodes=1  --node_rank=0 --master_addr=localhost \
#     run_class_finetuning.py \
#     --model vit_large_patch16_224 \
#     --data_set Kinetics-400 \
#     --nb_classes 16 \
#     --data_path ${DATA_PATH} \
#     --finetune ${MODEL_PATH} \
#     --log_dir ${OUTPUT_DIR} \
#     --output_dir ${OUTPUT_DIR} \
#     --batch_size 2 \
#     --num_sample 1 \
#     --input_size 224 \
#     --short_side_size 224\
#     --save_ckpt_freq 5 \
#     --num_frames 16 \
#     --sampling_rate 8 \
#     --opt lion \
#     --opt_betas 0.9 0.999 \
#     --warmup_epochs 5 \
#     --epochs 35 \
#     --test_num_segment 10 \
#     --test_num_crop 1 \
#     --lr 2e-3 \
#     --view "dash_0" \
#     --layer_decay 0.75 \
#     --dist_eval \
#     --fc_drop_rate 0.5 \

# OUTPUT_DIR='../../model_zoo/k400_intervideo_pretrain_large_patch16_224_frame_16x8_tube_mask_ratio_0.9_e1600/K400_init_dash_view_baseline_newgt_fold1'
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node=5 \
#     --master_port 12320 --nnodes=1  --node_rank=0 --master_addr=localhost \
#     run_class_finetuning.py \
#     --model vit_large_patch16_224 \
#     --data_set Kinetics-400 \
#     --nb_classes 16 \
#     --data_path ${DATA_PATH} \
#     --finetune ${MODEL_PATH} \
#     --log_dir ${OUTPUT_DIR} \
#     --output_dir ${OUTPUT_DIR} \
#     --batch_size 2 \
#     --num_sample 1 \
#     --input_size 224 \
#     --short_side_size 224\
#     --save_ckpt_freq 5 \
#     --num_frames 16 \
#     --sampling_rate 8 \
#     --opt lion \
#     --opt_betas 0.9 0.999 \
#     --warmup_epochs 5 \
#     --epochs 35 \
#     --test_num_segment 10 \
#     --test_num_crop 1 \
#     --lr 2e-3 \
#     --view "dash_1" \
#     --layer_decay 0.75 \
#     --dist_eval \
#     --fc_drop_rate 0.5 \




# OUTPUT_DIR='../../model_zoo/k400_intervideo_pretrain_large_patch16_224_frame_16x8_tube_mask_ratio_0.9_e1600/K400_init_dash_view_baseline_newgt_fold2'
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node=5 \
#     --master_port 12320 --nnodes=1  --node_rank=0 --master_addr=localhost \
#     run_class_finetuning.py \
#     --model vit_large_patch16_224 \
#     --data_set Kinetics-400 \
#     --nb_classes 16 \
#     --data_path ${DATA_PATH} \
#     --finetune ${MODEL_PATH} \
#     --log_dir ${OUTPUT_DIR} \
#     --output_dir ${OUTPUT_DIR} \
#     --batch_size 2 \
#     --num_sample 1 \
#     --input_size 224 \
#     --short_side_size 224\
#     --save_ckpt_freq 5 \
#     --num_frames 16 \
#     --sampling_rate 8 \
#     --opt lion \
#     --opt_betas 0.9 0.999 \
#     --warmup_epochs 5 \
#     --epochs 35 \
#     --test_num_segment 10 \
#     --test_num_crop 1 \
#     --lr 2e-3 \
#     --view "dash_2" \
#     --layer_decay 0.75 \
#     --dist_eval \
#     --fc_drop_rate 0.5 \



# OUTPUT_DIR='../../model_zoo/k400_intervideo_pretrain_large_patch16_224_frame_16x8_tube_mask_ratio_0.9_e1600/K400_init_dash_view_baseline_newgt_fold3'
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node=5 \
#     --master_port 12320 --nnodes=1  --node_rank=0 --master_addr=localhost \
#     run_class_finetuning.py \
#     --model vit_large_patch16_224 \
#     --data_set Kinetics-400 \
#     --nb_classes 16 \
#     --data_path ${DATA_PATH} \
#     --finetune ${MODEL_PATH} \
#     --log_dir ${OUTPUT_DIR} \
#     --output_dir ${OUTPUT_DIR} \
#     --batch_size 2 \
#     --num_sample 1 \
#     --input_size 224 \
#     --short_side_size 224\
#     --save_ckpt_freq 5 \
#     --num_frames 16 \
#     --sampling_rate 8 \
#     --opt lion \
#     --opt_betas 0.9 0.999 \
#     --warmup_epochs 5 \
#     --epochs 35 \
#     --test_num_segment 10 \
#     --test_num_crop 1 \
#     --lr 2e-3 \
#     --view "dash_3" \
#     --layer_decay 0.75 \
#     --dist_eval \
#     --fc_drop_rate 0.5 \


# OUTPUT_DIR='../../model_zoo/k400_intervideo_pretrain_large_patch16_224_frame_16x8_tube_mask_ratio_0.9_e1600/K400_init_dash_view_baseline_newgt_fold4'
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node=5 \
#     --master_port 12320 --nnodes=1  --node_rank=0 --master_addr=localhost \
#     run_class_finetuning.py \
#     --model vit_large_patch16_224 \
#     --data_set Kinetics-400 \
#     --nb_classes 16 \
#     --data_path ${DATA_PATH} \
#     --finetune ${MODEL_PATH} \
#     --log_dir ${OUTPUT_DIR} \
#     --output_dir ${OUTPUT_DIR} \
#     --batch_size 2 \
#     --num_sample 1 \
#     --input_size 224 \
#     --short_side_size 224\
#     --save_ckpt_freq 5 \
#     --num_frames 16 \
#     --sampling_rate 8 \
#     --opt lion \
#     --opt_betas 0.9 0.999 \
#     --warmup_epochs 5 \
#     --epochs 35 \
#     --test_num_segment 10 \
#     --test_num_crop 1 \
#     --lr 2e-3 \
#     --view "dash_4" \
#     --layer_decay 0.75 \
#     --dist_eval \
#     --fc_drop_rate 0.5 \


OUTPUT_DIR='../../model_zoo/k400_intervideo_pretrain_large_patch16_224_frame_16x8_tube_mask_ratio_0.9_e1600/exp_dash_2_ema/'
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=5 \
    --master_port 12320 --nnodes=1  --node_rank=0 --master_addr=localhost \
    run_class_finetuning.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 2 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 2 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.999 \
    --warmup_epochs 5 \
    --epochs 20 \
    --test_num_segment 10 \
    --test_num_crop 1 \
    --lr 1e-3 \
    --view "dash_2_clean" \
    --layer_decay 0.75 \
    --dist_eval \
    --model_ema \
  


OUTPUT_DIR='../../model_zoo/k400_intervideo_pretrain_large_patch16_224_frame_16x8_tube_mask_ratio_0.9_e1600/exp_right_2_ema/'
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=5 \
    --master_port 12320 --nnodes=1  --node_rank=0 --master_addr=localhost \
    run_class_finetuning.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 2 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 2 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.999 \
    --warmup_epochs 5 \
    --epochs 20 \
    --test_num_segment 10 \
    --test_num_crop 1 \
    --lr 1e-3 \
    --view "right_2_clean" \
    --layer_decay 0.75 \
    --dist_eval \
    --model_ema \
  


OUTPUT_DIR='../../model_zoo/k400_intervideo_pretrain_large_patch16_224_frame_16x8_tube_mask_ratio_0.9_e1600/exp_rear_2_ema/'
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=5 \
    --master_port 12320 --nnodes=1  --node_rank=0 --master_addr=localhost \
    run_class_finetuning.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 2 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 2 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.999 \
    --warmup_epochs 5 \
    --epochs 20 \
    --test_num_segment 10 \
    --test_num_crop 1 \
    --lr 1e-3 \
    --view "rear_2_clean" \
    --layer_decay 0.75 \
    --dist_eval \
    --model_ema \
  