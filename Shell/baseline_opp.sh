# #!/bin/bash

# python main.py --dataset_name Opportunity++ --model_name primus --batch_size 32 
# echo "primus 끝."
# python main.py --dataset_name Opportunity++ --model_name mae --batch_size 12 
# echo "mae 끝."
# python main.py --dataset_name Opportunity++ --model_name imu2clip --batch_size 32 
# echo "primus 끝."
# python main.py --dataset_name Opportunity++ --model_name comodo --batch_size 12 
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
CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/mae/Opportunity++/last_bs=36_epoch=50.ckpt --supervision True
CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/primus/Opportunity++/last_bs=256_epoch=50.ckpt --supervision True
CUDA_VISIBLE_DEVICES=0,1,2,3, python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/method/Opportunity++/last_bs=256_epochs=20_original.ckpt --supervision True
CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/comodo/Opportunity++/last_bs=84*3_epoch=50.ckpt --supervision True
