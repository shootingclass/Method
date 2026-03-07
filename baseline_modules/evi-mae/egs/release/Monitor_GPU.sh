#!/bin/bash

# Configuration from finetune_hwu.sh
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
dataset=hwu
dataset_base_path=/mnt/hdd4tb/junho/HWU-USP_v2/data_processed_2s_window
data_train=${dataset_base_path}/motion_2_priority_test=18/linear_probe_train.json
data_val=${dataset_base_path}/motion_2_priority_test=18/linear_probe_test.json
label_csv=${dataset_base_path}/class_labels_indices.csv
pretrain_path=/home/jaemo/Method/checkpoints/evimae/HWU-USP/models/evi_model.299.pth
model=evi-mae-ft
bal=None
lr=5e-5
batch_size=1
epoch=200
lrscheduler_start=60
lrscheduler_decay=0.5
lrscheduler_step=60
wa=True
wa_start=40
wa_end=200
wa_num=12
freeze_base=False
head_lr=100
lr_adapt=False
base_lr=1

n_class=5
label_smooth=0.1
noise=True
freqm=48
timem=192
mixup=0

imu_plot_type=stft
imu_channel_num=6
imu_target_length=100
imu_plot_height=128
imu_patch_size=16 
imu_dataset_mean=/mnt/hdd4tb/junho/HWU-USP_v2/sensor_stats_6.npy
imu_dataset_std=/mnt/hdd4tb/junho/HWU-USP_v2/sensor_stats_6.npy

imu_enable_graph=False
imu_graph_net=gin
imu_two_stream=False

# small
imu_encoder_embed_dim=768
imu_encoder_depth=11
imu_encoder_num_heads=12

video_img_size=224
video_patch_size=16
video_encoder_num_classes=0
video_decoder_num_classes=1536
video_mlp_ratio=4
video_qkv_bias=True
video_masking_ratio=0.9

video_encoder_embed_dim=384
video_encoder_depth=12
video_encoder_num_heads=6 
video_decoder_embed_dim=192 
video_decoder_num_heads=3

metrics=acc

# Focus on sensor-video (multimodal) as it is likely the heaviest
encoder_type=sensor-video
ftmode=multimodal
exp_dir=${dataset_base_path}/evi-mae-exp/monitor_gpu/${encoder_type}

echo "============================================================"
echo "Starting GPU Monitor for encoder_type=${encoder_type}"
echo "============================================================"

mkdir -p $exp_dir

# Start GPU monitoring
echo "Time, GPU, Memory Used (MB)" > gpu_usage_log.csv
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -l 1 >> gpu_usage_log.csv &
MONITOR_PID=$!

# Run the training script with timeout
timeout 60s \
CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_CACHE_DISABLE=1 python -W ignore /home/jaemo/Method/baseline_modules/evi-mae/src/run_evimae_ft_seq.py --model ${model} --dataset ${dataset} \
    --data-train ${data_train} --data-val ${data_val} --exp-dir $exp_dir \
    --label-csv ${label_csv} --n_class ${n_class} \
    --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
    --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
    --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
    --label_smooth ${label_smooth} --noise ${noise} \
    --loss BCE --metrics ${metrics} --warmup True \
    --wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --wa_num ${wa_num} --lr_adapt ${lr_adapt} \
    --pretrain_path ${pretrain_path} --ftmode ${ftmode} \
    --freeze_base ${freeze_base} --head_lr ${head_lr} \
    --imu_plot_type ${imu_plot_type} --imu_plot_height ${imu_plot_height} --imu_patch_size ${imu_patch_size} \
    --imu_dataset_mean ${imu_dataset_mean} --imu_dataset_std ${imu_dataset_std} --imu_channel_num ${imu_channel_num} \
    --imu_target_length ${imu_target_length} \
    --video_img_size ${video_img_size} --video_patch_size ${video_patch_size} --video_encoder_num_classes ${video_encoder_num_classes} \
    --video_decoder_num_classes ${video_decoder_num_classes} --video_mlp_ratio ${video_mlp_ratio} --video_qkv_bias ${video_qkv_bias} \
    --video_encoder_embed_dim ${video_encoder_embed_dim} --video_encoder_depth ${video_encoder_depth} \
    --video_encoder_num_heads ${video_encoder_num_heads} --video_decoder_embed_dim ${video_decoder_embed_dim} \
    --video_decoder_num_heads ${video_decoder_num_heads} --video_masking_ratio ${video_masking_ratio} \
    --imu_encoder_embed_dim ${imu_encoder_embed_dim} --imu_encoder_depth ${imu_encoder_depth} \
    --imu_encoder_num_heads ${imu_encoder_num_heads} --num-workers 8 \
    --imu_enable_graph ${imu_enable_graph} --imu_graph_net ${imu_graph_net} --base_lr ${base_lr} \
    --imu_two_stream ${imu_two_stream}

# Stop monitoring
kill $MONITOR_PID
echo "Done. Check gpu_usage_log.csv"
