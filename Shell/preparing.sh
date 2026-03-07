python main.py \
    --dataset_name Opportunity++ \
    --model_name method \
    --project_name ECCV_Pretraining \
    --epochs 21 \
    --lr 1e-4 \
    --batch_size 32 \
    --num_frames 20 \
    --embedding_dim 512 \
    --threshold_epoch -1 \
    --centroid_threshold 0.75 \
    --video_classifier_epoch 11 \
    --bad_correction_epoch 10 \
    --save_stage_cache

python main.py \
    --dataset_name HWU-USP \
    --model_name method \
    --project_name ECCV_Pretraining \
    --epochs 21 \
    --lr 1e-4 \
    --batch_size 32 \
    --num_frames 20 \
    --embedding_dim 512 \
    --threshold_epoch -1 \
    --centroid_threshold 0.75 \
    --video_classifier_epoch 11 \
    --bad_correction_epoch 10 \
    --save_stage_cache