
DATA_PATH='data/A1_clip'
MODEL_PATH='/path/to/pretrained_models/vit_l_hybrid_pt_800e_k700_ft.pth'
OUTPUT_DIR='./checkpoints/recog_dash_0/'
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
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
    --epochs 35 \
    --test_num_segment 10 \
    --test_num_crop 1 \
    --lr 1e-3 \
    --view "dash_0" \
    --layer_decay 0.75 \
    --dist_eval \
  
OUTPUT_DIR='./checkpoints/recog_right_0/'
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
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
    --epochs 35 \
    --test_num_segment 10 \
    --test_num_crop 1 \
    --lr 1e-3 \
    --view "right_0" \
    --layer_decay 0.75 \
    --dist_eval \

OUTPUT_DIR='./checkpoints/recog_rear_0/'
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
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
    --epochs 35 \
    --test_num_segment 10 \
    --test_num_crop 1 \
    --lr 1e-3 \
    --view "rear_0" \
    --layer_decay 0.75 \
    --dist_eval \