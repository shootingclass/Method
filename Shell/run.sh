# # CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset_name Opportunity++ --threshold_epoch 0 --video_classifier_epoch 0 --bad_correction_epoch 0 --batch_size 32 --momentum_m 0.999  --lambda_hard 3.0 --contrastive_temp 0.10 --damp_warmup_epochs 0 --num_workers 0
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset_name HWU-USP --batch_size 32 --momentum_m 0.999  --lambda_hard 3.0 --contrastive_temp 0.10 --threshold_epoch 0 --video_classifier_epoch 0 --bad_correction_epoch 0 --damp_warmup_epochs 0 --num_workers 0 --epochs 50

# # CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset_name Opportunity++ --batch_size 32 --momentum_m 0.999  --lambda_hard 3.0 --contrastive_temp 0.10 --damp_warmup_epochs 0 
# # CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --dataset_name HWU-USP --batch_size 32 --momentum_m 0.999  --lambda_hard 3.0 --contrastive_temp 0.10 --damp_warmup_epochs 0 --epochs 100
# python main.py \
#     --dataset_name Opportunity++ \
#     --model_name comodo \
#     --project_name ECCV_Pretraining \
#     --epochs 50 \
#     --lr 1e-4 \
#     --batch_size 8 \
#     --embedding_dim 512 \
#     --threshold_epoch -1 \
#     --centroid_threshold 0.75 \

# python main.py \
#     --dataset_name HWU-USP \
#     --model_name comodo \
#     --project_name ECCV_Pretraining \
#     --epochs 50 \
#     --lr 1e-4 \
#     --batch_size 8 \
#     --embedding_dim 512 \
#     --threshold_epoch -1 \
#     --centroid_threshold 0.75 \

# python main.py \
#     --dataset_name HWU-USP \
#     --model_name method \
#     --project_name ECCV_Pretraining \
#     --epochs 50 \
#     --lr 1e-4 \
#     --batch_size 48 \
#     --embedding_dim 512 \
#     --threshold_epoch 0 \
#     --centroid_threshold 0.75 \
#     --video_classifier_epoch 0 \
#     --bad_correction_epoch 0 \
#     --use_flow

# python main.py \
#     --dataset_name HWU-USP \
#     --model_name method \
#     --project_name ECCV_Pretraining \
#     --epochs 50 \
#     --lr 1e-4 \
#     --batch_size 64 \
#     --embedding_dim 512 \
#     --threshold_epoch 0 \
#     --centroid_threshold 0.75 \
#     --video_classifier_epoch 0 \
#     --bad_correction_epoch 0 

# python main.py \
#     --dataset_name HWU-USP \
#     --model_name method \
#     --project_name ECCV_Pretraining \
#     --epochs 50 \
#     --lr 1e-4 \
#     --batch_size 64 \
#     --num_frames 20 \
#     --embedding_dim 512 \
#     --threshold_epoch 0 \
#     --centroid_threshold 0.75 \
#     --video_classifier_epoch 0 \
#     --bad_correction_epoch 0 \
#     --use_flow

python main.py \
    --dataset_name Opportunity++ \
    --model_name method \
    --project_name ECCV_Pretraining \
    --epochs 50 \
    --lr 1e-4 \
    --batch_size 8 \
    --embedding_dim 512 \
    --threshold_epoch 0 \
    --centroid_threshold 0.75 \
    --video_classifier_epoch 0 \
    --bad_correction_epoch 0 \
    --use_flow
# python main.py \
#     --dataset_name HWU-USP \
#     --model_name primus \
#     --project_name ECCV_Pretraining \
#     --epochs 50 \
#     --lr 1e-4 \
#     --batch_size 32 \
#     --embedding_dim 512 \
#     --threshold_epoch -1 \
#     --centroid_threshold 0.75 \
#     --video_classifier_epoch 0 \
#     --bad_correction_epoch 0 
