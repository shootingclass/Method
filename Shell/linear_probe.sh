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
echo "linear probe.sh 시작"
# python main.py --dataset_name HWU-USP --model_name primus --batch_size 16
# echo "primus 끝"
# python main.py --dataset_name HWU-USP --model_name imu2clip --batch_size 16
# echo "imu2clip 끝"
# python main.py --dataset_name HWU-USP --model_name comodo --batch_size 8
# CUDA_VISIBLE_DEVICES=0 python linear_probe.py --batch_size 1 --checkpoint_path /home/jaemo/Method/checkpoints/imu2clip/HWU-USP/last_bs=64_epoch=50.ckpt
# for epoch in 1 3 5 10 15 19
# for epoch in 99:
# do
#   python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/method/Opportunity++/manual_epochs/manual_epoch_15.ckpt --linear_epochs 20
# done
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset_name Opportunity++ --batch_size 64 --threshold_epoch 0 --video_classifier_epoch 0 --bad_correction_epoch 0 --momentum_m 0.999 --lambda_hard 3.0 --contrastive_temp 0.10 --epochs 16
# python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/method/Opportunity++/manual_epochs_all_confident/manual_epoch_15.ckpt --linear_epochs 20
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_probe.py \
#       --batch_size 64 \
#       --checkpoint_path /home/jaemo/Method/checkpoints/method/Opportunity++/last_bs=256_epochs=15_all_confident_again.ckpt \
#       --linear_epochs 20

# for epoch in 15 10
# do

#   CUDA_VISIBLE_DEVICES=0 python linear_probe.py \
#     --batch_size 1 \
#     --checkpoint_path /home/jaemo/Method/checkpoints/method/HWU-USP/manual_epochs_all_confident/manual_epoch_${epoch}.ckpt \
#     --linear_epochs 20
# done
# echo "linear probe 끝"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset_name Opportunity++ --batch_size 64 --threshold_epoch 0 --video_classifier_epoch 0 --bad_correction_epoch 0 --momentum_m 0.999 --lambda_hard 3.0 --contrastive_temp 0.10 --epochs 50

# python main.py --dataset_name HWU-USP --model_name mae --batch_size 8
# echo "mae 끝 HWU 끝"
# CUDA_VISIBLE_DEVICES=0 python linear_probe.py --batch_size 1 --checkpoint_path /home/jaemo/Method/checkpoints/primus/HWU-USP/last_bs=64_epoch=50.ckpt
# CUDA_VISIBLE_DEVICES=0 python linear_probe.py --batch_size 1 --checkpoint_path /home/jaemo/Method/checkpoints/mae/HWU-USP/last_bs=32_epoch=50.ckpt
# CUDA_VISIBLE_DEVICES=0 python linear_probe.py --batch_size 1 --checkpoint_path /home/jaemo/Method/checkpoints/comodo/HWU-USP/last_bs=84*3_epoch=50.ckpt
# 시작 전에는 폴더 저장!!
## 중간 중간 폴더 저장해주면 좋음!

# python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/method/Opportunity++/manual_epochs_transformer/manual_epoch_25.ckpt
# #   --linear_epochs 20
# python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/method/HWU-USP/manual_epochs_transformer/manual_epoch_24.ckpt
#   --linear_epochs 20
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset_name Opportunity++ --batch_size 32 --threshold_epoch 0 --video_classifier_epoch 0 --bad_correction_epoch 0 --momentum_m 0.999  --lambda_hard 3.0 --contrastive_temp 0.10 --damp_warmup_epochs 0 --num_workers 0
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset_name HWU-USP --batch_size 32 --threshold_epoch 0 --video_classifier_epoch 0 --bad_correction_epoch 0 --momentum_m 0.999  --lambda_hard 3.0 --contrastive_temp 0.10 --damp_warmup_epochs 0 --num_workers 0
# for model in primus imu2clip comodo mae
model=method
for dataset_name in Opportunity++
do
    for encoder_type in video sensor-video
    do
        if [ "$model" == "method" ]; then   
            python linear_probe.py --batch_size 4 --checkpoint_path /home/jaemo/Method/checkpoints/${model}/${dataset_name}/mvit_flow.ckpt --encoder_type ${encoder_type} --use_flow
            # python linear_probe.py --batch_size 4 --checkpoint_path /home/jaemo/Method/checkpoints/${model}/${dataset_name}/noFlow.ckpt --encoder_type ${encoder_type}
            # python linear_probe.py --batch_size 4 --checkpoint_path /home/jaemo/Method/checkpoints/${model}/${dataset_name}/crossModal.ckpt --encoder_type ${encoder_type} --use_flow
        else
            python linear_probe.py --batch_size 4 --checkpoint_path /home/jaemo/Method/checkpoints/${model}/${dataset_name}/${model}.ckpt --encoder_type ${encoder_type}
        fi
        echo "${dataset_name} ${encoder_type} linear probe 끝"
    done
done
# python linear_probe.py --batch_size 4 --checkpoint_path /home/jaemo/Method/checkpoints/${model}/Opportunity++/crossModal.ckpt --encoder_type sensor-video --use_flow
# python linear_probe.py --batch_size 4 --checkpoint_path /home/jaemo/Method/checkpoints/${model}/Opportunity++/noFlow.ckpt --encoder_type sensor
# python linear_probe.py --batch_size 4 --checkpoint_path /home/jaemo/Method/checkpoints/${model}/Opportunity++/crossModal.ckpt --encoder_type video --use_flow

# python linear_probe.py --batch_size 4 --checkpoint_path /home/jaemo/Method/checkpoints/${model}/HWU-USP/noFlow_diff.ckpt --encoder_type video --linear_epochs 50
# python linear_probe.py --batch_size 4 --checkpoint_path /home/jaemo/Method/checkpoints/${model}/HWU-USP/crossModal.ckpt --encoder_type video --use_flow --probe_mode lstm
# python linear_probe.py --batch_size 4 --checkpoint_path /home/jaemo/Method/checkpoints/${model}/HWU-USP/noFlow.ckpt --encoder_type sensor-video
# python linear_probe.py --batch_size 4 --checkpoint_path /home/jaemo/Method/checkpoints/${model}/HWU-USP/noFlow.ckpt --encoder_type sensor
# python linear_probe.py --batch_size 4 --checkpoint_path /home/jaemo/Method/checkpoints/${model}/HWU-USP/crossModal.ckpt --encoder_type sensor-video --use_flow
# python linear_probe.py --batch_size 4 --checkpoint_path /home/jaemo/Method/checkpoints/${model}/HWU-USP/crossModal.ckpt --encoder_type sensor --use_flow


# CUDA_VISIBLE_DEVICES=0 python linear_probe.py \
#   --batch_size 1 \
#   --checkpoint_path /home/jaemo/Method/checkpoints/primus/HWU-USP/last_bs=64_epoch=15.ckpt \
#   --linear_epochs 20


# CUDA_VISIBLE_DEVICES=0 python linear_probe.py \
#   --batch_size 1 \
#   --checkpoint_path /home/jaemo/Method/checkpoints/imu2clip/HWU-USP/last_bs=64_epoch=15.ckpt \
#   --linear_epochs 20


# CUDA_VISIBLE_DEVICES=0 python linear_probe.py \
#   --batch_size 1 \
#   --checkpoint_path /home/jaemo/Method/checkpoints/comodo/HWU-USP/last_bs=84*3_epoch=15.ckpt \
#   --linear_epochs 20


# CUDA_VISIBLE_DEVICES=0 python linear_probe.py \
#   --batch_size 1 \
#   --checkpoint_path /home/jaemo/Method/checkpoints/mae/HWU-USP/last.ckpt \
#   --linear_epochs 20

# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset_name Opportunity++ --batch_size 64 --threshold_epoch 0 --video_classifier_epoch 0 --bad_correction_epoch 0 --momentum_m 0.999 --lambda_hard 3.0 --contrastive_temp 0.10 --epochs 16