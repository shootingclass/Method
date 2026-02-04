# #!/bin/bash
# 시작 전에는 폴더 저장!!
## 중간 중간 폴더 저장해주면 좋음!
# python main.py --dataset_name HWU-USP --model_name primus --batch_size 16 --epochs 16

# CUDA_VISIBLE_DEVICES=0 python linear_probe.py --batch_size 1 --checkpoint_path /home/jaemo/Method/checkpoints/primus/HWU-USP/last_bs=64_epoch=15.ckpt --linear_epoch 20
# CUDA_VISIBLE_DEVICES=0 python linear_probe.py --batch_size 1 --checkpoint_path /home/jaemo/Method/checkpoints/primus/HWU-USP/last_bs=64_epoch=50.ckpt --linear_epoch 20

# CUDA_VISIBLE_DEVICES=0 python linear_probe.py --batch_size 1 --checkpoint_path /home/jaemo/Method/checkpoints/mae/HWU-USP/last_bs=32_epoch=50.ckpt --linear_epoch 20
for epoch in 10 
do
  CUDA_VISIBLE_DEVICES=0 python linear_probe.py --batch_size 1 --checkpoint_path /home/jaemo/Method/checkpoints/method/HWU-USP/manual_epochs_noRefine+infoNce/manual_epoch_${epoch}.ckpt --linear_epoch 20
done