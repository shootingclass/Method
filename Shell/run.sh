# CUDA_VISIBLE_DEVICES=1,2,3 python /home/junho/Method/main.py
#!/bin/bash
echo "실험 1 시작..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/primus/Opportunity++/last_bs=256_epoch=50.ckpt
echo "실험 1 완료."
CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/imu2clip/Opportunity++/last_bs=256_epoch=50.ckpt
echo "실험 2 완료."
CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_probe.py --batch_size 64 --checkpoint_path /home/jaemo/Method/checkpoints/method/Opportunity++/last.ckpt
echo "실험 3 완료."
CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_probe.py --batch_size 12 --checkpoint_path /home/jaemo/Method/checkpoints/comodo/Opportunity++/last_bs=84*3_epoch=50.ckpt