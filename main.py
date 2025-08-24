import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm 
import wandb
import numpy as np
import random
import itertools

# 분산 학습 라이브러리
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# --- 사용자 정의 모듈 임포트 ---
from dataset import VideoSensorDataset, SensorTransform, ClipConsistentTransforms
from model import  SensorModel, VisionModel
from utils import (
    train_one_epoch, 
    calculate_sensor_stats, save_stats, load_stats
)


####################################################################


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


####################################################################


def main():

    # --- 분산 학습 설정 ---
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    set_random_seed(42)

    if rank == 0:
        print(f"Using {world_size} GPUs for distributed training.")

        # wandb.init(
        #     project="Method_Test",
        #     name=f"Test1",
        # )
    
    set_random_seed(42)

    # ==================================================================
    # 1. 하이퍼파라미터 및 설정 정의
    # ==================================================================
    DATA_ROOT = "/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window/"
    JSON_TRAIN_PATH = "/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window/noToggle/pretrain.json"
    STATS_FILE_PATH = '/home/junho/Method/sensor_stats/sensor_stats.npy' # 센서 데이터 통계 파일 경로
    NUM_FRAMES = 16
    BATCH_SIZE = 4
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4

    # 시각화 결과물을 저장할 폴더 이름
    output_dir = "Visualization/transformed_video"
    
    # 폴더가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)

    # ==================================================================


    # ==================================================================
    # 2. 프레임 전처리(Transform) 정의
    # ==================================================================
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]    
    
    # 학습(Training)용 변환
    train_transform = ClipConsistentTransforms(
        size=(224, 224),
        mean=clip_mean,
        std=clip_std
    )

    # ==================================================================


    # ==================================================================
    # 3. 센서 데이터 전처리 준비
    # ==================================================================
    if os.path.exists(STATS_FILE_PATH):
        # 파일이 존재하면, 통계치를 불러옵니다.
        stats = load_stats(STATS_FILE_PATH)

    else:
        # 파일이 없으면, 통계치를 계산하고 저장합니다.
        print(f"Statistics file not found. Calculating for the first time...")
        
        # 통계 계산용 임시 데이터셋 생성 (전처리 없음)
        temp_train_dataset = VideoSensorDataset(
            json_path=JSON_TRAIN_PATH,
            data_root=DATA_ROOT,
            num_frames=NUM_FRAMES,
            transform=train_transform,
            sensor_transform=None # 센서 변환 없음
        )
        stats = calculate_sensor_stats(temp_train_dataset)
        save_stats(stats, STATS_FILE_PATH)

    # 불러오거나 계산된 통계치를 사용하여 SensorTransform 객체 생성
    sensor_preprocessor = SensorTransform(target_len=128, mean=stats['mean'], std=stats['std'])
    # ==================================================================


    # ==================================================================
    # 4. 데이터셋(Dataset) 및 데이터로더(DataLoader) 생성
    # ==================================================================
    train_dataset = VideoSensorDataset(
        json_path=JSON_TRAIN_PATH,
        data_root=DATA_ROOT,
        num_frames=NUM_FRAMES,
        transform=train_transform,
        sensor_transform=sensor_preprocessor # 센서 전처리 적용
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # DistributedSampler가 셔플링을 담당합니다.
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler
    )

    print(f"Train dataset size: {len(train_dataset)}")
    
    # ==================================================================


    # ==================================================================
    # 5. 모델, 손실 함수, 옵티마이저 정의
    # ==================================================================
    video_model = VisionModel(image_size=224).to(DEVICE)
    sensor_model = SensorModel(sensor_channels=97).to(DEVICE)

    video_model = DDP(video_model, device_ids=[local_rank], find_unused_parameters=True)
    sensor_model = DDP(sensor_model, device_ids=[local_rank], find_unused_parameters=True)

    parameters = itertools.chain(video_model.parameters(), sensor_model.parameters())
    optimizer = optim.AdamW(parameters, lr=LEARNING_RATE)
    # ==================================================================


    # ==================================================================
    # 6. 데이터 로더 테스트 (실제 학습 전 확인용)
    # ==================================================================
    print("\n--- Testing Dataloader ---")
    
    try:
        videos, sensors, labels = next(iter(train_loader))
        print(f"Video batch shape: {videos.shape}")
        print(f"Sensor batch shape: {sensors.shape}")
        print(f"Labels batch shape: {labels.shape}")
        
        # 센서 데이터의 평균과 표준편차 확인 (전처리 확인용)
        # 배치의 첫 번째 센서 데이터에 대해 계산
        sensor_sample = sensors[0].numpy()
        print(f"Sensor sample mean (post-transform): {np.mean(sensor_sample, axis=1)}")
        print(f"Sensor sample std (post-transform): {np.std(sensor_sample, axis=1)}")
        
        print("Dataloader test successful!")
        
    except Exception as e:
        print(f"Error during dataloader test: {e}")
    # ==================================================================


    # ==================================================================
    # 7. 학습 및 검증 루프 (Training & Validation Loop)
    # ==================================================================

    if rank == 0:
        print("\n--- Starting Training ---")

    for epoch in range(EPOCHS):
        train_loader.sampler.set_epoch(epoch)  # 중요: 에포크마다 샘플러 상태를 업데이트합니다.

        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # --- 1. 학습 단계 ---
        # train_one_epoch 함수에 device 변수를 전달합니다.
        train_loss = train_one_epoch(video_model, sensor_model, train_loader, optimizer,device, epoch, output_dir, rank)

        # train_one_epoch 함수가 loss를 모든 프로세스에 브로드캐스팅하지 않는다면,
        # 아래 코드는 각 프로세스별 loss를 출력할 수 있습니다.
        # 모든 프로세스의 평균 loss를 보려면 추가적인 동기화 코드가 필요합니다.
        if rank == 0:
            print(f"[Train] Loss: {train_loss:.4f}")

    if rank == 0:
        print("\n--- Training Complete ---")

    # ==================================================================

    # 분산 처리 관련 리소스 정리
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
