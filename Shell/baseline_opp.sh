# #!/bin/bash

# python main.py --dataset_name Opportunity++ --model_name primus --batch_size 32 --epochs 50
# echo "primus 끝."
# python main.py --dataset_name Opportunity++ --model_name mae --batch_size 12 --epochs 50
# echo "mae 끝."
# python main.py --dataset_name Opportunity++ --model_name imu2clip --batch_size 32 --epochs 50
# echo "imu2clip 끝."
# python main.py --dataset_name Opportunity++ --model_name comodo --batch_size 12 --epochs 50
# echo "comodo 끝."

# for model in mae
# do
#     python linear_probe.py --batch_size 16 --checkpoint_path /home/jaemo/Method/checkpoints/${model}/Opportunity++/${model}.ckpt 
#     python linear_probe.py --batch_size 32 --checkpoint_path /home/jaemo/Method/checkpoints/${model}/Opportunity++/${model}.ckpt 
#     echo "${model} linear probe 끝"
# done
# python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/mae/Opportunity++/last_bs=32_epoch_15_evi.ckpt
# python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/mae/Opportunity++/last_bs=36_epoch=20.ckpt
# # python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/mae/Opportunity++/last_bs=36_epoch=50.ckpt
for model in comodo imu2clip primus mae method
do
    python linear_probe.py --batch_size 1 --checkpoint_path /home/jaemo/Method/checkpoints/${model}/HWU-USP/${model}.ckpt --linear_epochs 20
    echo "${model} linear probe 끝"
done

# python main.py --dataset_name HWU-USP --model_name primus --batch_size 32 --epochs 50
# echo "primus 끝."
# python main.py --dataset_name HWU-USP --model_name mae --batch_size 12 --epochs 50
# echo "mae 끝."
# python main.py --dataset_name HWU-USP --model_name imu2clip --batch_size 32 --epochs 50
# echo "primus 끝."
# python main.py --dataset_name HWU-USP --model_name comodo --batch_size 12 --epochs 50
# echo "mae 끝."



# CUDA_VISIBLE_DEVICES=0,1,2,3, python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/method/Opportunity++/last_bs=256_epochs=20_original.ckpt
# CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/comodo/Opportunity++/last_bs=84*3_epoch=50.ckpt
# CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/imu2clip/Opportunity++/last_bs=256_epoch=50.ckpt
# CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/mae/Opportunity++/last_bs=36_epoch=50.ckpt
# CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/primus/Opportunity++/last_bs=256_epoch=50.ckpt

echo "opp linear probe end"


echo "main"

# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset_name Opportunity++ --batch_size 64 --threshold_epoch 0 --video_classifier_epoch 0 --bad_correction_epoch 0 --momentum_m 0.75  --lambda_hard 2.0 --contrastive_temp 0.10 --damp_warmup_epochs 0
echo "main 끝"

# CUDA_VISIBLE_DEVICES=0,1,2,3, python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/method/Opportunity++/last-v3.ckpt
# CUDA_VISIBLE_DEVICES=0,1,2,3, python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/method/Opportunity++/last_bs=256_epochs=20_original.ckpt --linear_epochs 100

# CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/imu2clip/Opportunity++/last_bs=256_epoch=50.ckpt --supervision True
# CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/mae/Opportunity++/last_bs=36_epoch=50.ckpt --supervision True
# CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/primus/Opportunity++/last_bs=256_epoch=50.ckpt --supervision True
# CUDA_VISIBLE_DEVICES=0,1,2,3, python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/method/Opportunity++/last_bs=256_epochs=20_original.ckpt --supervision True
# CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/comodo/Opportunity++/last_bs=84*3_epoch=50.ckpt --supervision True
