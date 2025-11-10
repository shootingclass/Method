# # CUDA_VISIBLE_DEVICES=1,2,3 python /home/junho/Method/main.py
# #!/bin/bash


# python main.py --dataset_name Opportunity++ --model_name primus --batch_size 32 
# echo "primus 끝."
# python main.py --dataset_name Opportunity++ --model_name mae --batch_size 12 
# echo "mae 끝."
# python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/method/Opportunity++/last-v46.ckpt
# CUDA_VISIBLE_DEVICES=0 python linear_probe.py --batch_size 1 --checkpoint_path /home/jaemo/Method/checkpoints/imu2clip/HWU-USP/last_bs=64_epoch=50.ckpt
# CUDA_VISIBLE_DEVICES=0 python linear_probe.py --batch_size 1 --checkpoint_path /home/jaemo/Method/checkpoints/mae/HWU-USP/last_bs=32_epoch=50.ckpt

# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset_name HWU-USP --batch_size 64 --threshold_epoch 0 --video_classifier_epoch 0 --bad_correction_epoch 0 --momentum_m 0.75  --lambda_hard 2.0 --contrastive_temp 0.10 --damp_warmup_epochs 0

# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset_name Opportunity++ --batch_size 64 --threshold_epoch 0 --video_classifier_epoch 0 --bad_correction_epoch 0 --momentum_m 0.75  --lambda_hard 2.0 --contrastive_temp 0.10 --damp_warmup_epochs 0
echo "main 끝"
# python main.py --dataset_name HWU-USP --model_name primus --batch_size 16
# echo "primus 끝"
# python main.py --dataset_name HWU-USP --model_name imu2clip --batch_size 16
# echo "imu2clip 끝"
# python main.py --dataset_name HWU-USP --model_name comodo --batch_size 8
CUDA_VISIBLE_DEVICES=0 python linear_probe.py --batch_size 1 --checkpoint_path /home/jaemo/Method/checkpoints/imu2clip/HWU-USP/last_bs=64_epoch=50.ckpt
# for epoch in 1 3 5 10 15 19
for epoch in 3 5 10 15 19
do
  CUDA_VISIBLE_DEVICES=0 python linear_probe.py \
    --batch_size 1 \
    --checkpoint_path /home/jaemo/Method/checkpoints/method/HWU-USP/manual_epoch_${epoch}.ckpt
done
echo "comodo 끝"
# python main.py --dataset_name HWU-USP --model_name mae --batch_size 8
# echo "mae 끝 HWU 끝"
CUDA_VISIBLE_DEVICES=0 python linear_probe.py --batch_size 1 --checkpoint_path /home/jaemo/Method/checkpoints/primus/HWU-USP/last_bs=64_epoch=50.ckpt
CUDA_VISIBLE_DEVICES=0 python linear_probe.py --batch_size 1 --checkpoint_path /home/jaemo/Method/checkpoints/mae/HWU-USP/last_bs=32_epoch=50.ckpt
CUDA_VISIBLE_DEVICES=0 python linear_probe.py --batch_size 1 --checkpoint_path /home/jaemo/Method/checkpoints/comodo/HWU-USP/last_bs=84*3_epoch=50.ckpt
