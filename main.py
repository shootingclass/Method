import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm # tqdm 라이브러리 임포트 추가
import wandb
import numpy as np
import random
import argparse

# --- 사용자 정의 모듈 임포트 ---
from dataset import VideoSensorDataset, SensorTransform
from model import Clip4ClipVisionModel, MW2StackRNNPooling, ViTWithCAM
from clustering_model import HybridClusteringModule, initialize_prototypes
from utils import (
    train_one_epoch_with_cam, 
    validation_one_epoch_with_cam,
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


def main(args):
    
    set_random_seed(42)

    wandb.init(
        project="Method_Test",  # 원하는 프로젝트 이름으로 변경 가능
        name=f"CNN_SpatialPooling_Resize_HorizontalFlip_Door1",
        )

    # ==================================================================
    # 1. 하이퍼파라미터 및 설정 정의
    # ==================================================================
    DATA_ROOT = "/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window/"
    JSON_TRAIN_PATH = "/home/jaemo/jaemo_Opportunity++/custom_train.json"
    JSON_VAL_PATH = "/home/jaemo/jaemo_Opportunity++/custom_val.json" 
    STATS_FILE_PATH = '/home/jaemo/Method/sensor_stats/sensor_stats.npy' # 센서 데이터 통계 파일 경로
    NUM_CLASSES = args.num_classes
    NUM_FRAMES = 16
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    EMBEDDING_DIM = args.embedding_dim
    NUM_SENSORS = args.num_sensors
    ALPHA_FIXED = args.alpha_fixed
    THRESHOLD_EPOCH = args.threshold_epoch

    print(f"Using device: {DEVICE}")
    print(f"Number of classes: {NUM_CLASSES}")

    # 시각화 결과물을 저장할 폴더 이름
    output_dir = "cam_visualizations_Door2_Augmentation_ver2"
    
    # 폴더가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)

    # ==================================================================


    # ==================================================================
    # 2. 프레임 전처리(Transform) 정의
    # ==================================================================
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]    
    
    # 훈련(Training)용 변환
    train_transform = transforms.Compose([
        
        # RandomResizedCrop 제거 -> Resize로 변경
        # 프레임 전체의 정보를 보존하여 모델이 스스로 중요한 위치를 찾도록 함
        transforms.Resize(size=(224, 224), antialias=True),

        # 수평 뒤집기 (50% 확률)
        transforms.RandomHorizontalFlip(p=0.5),  
        
        # 배경 편향 방지를 위한 데이터 증강 추가
        # ColorJitter: 배경의 색감/조명에 대한 의존도를 낮춤
        # transforms.ColorJitter(
        #     brightness=0.5, # 기존 0.4 -> 0.5
        #     contrast=0.5,   # 기존 0.4 -> 0.5
        #     saturation=0.5, # 기존 0.4 -> 0.5
        #     hue=0.2         # 기존 0.1 -> 0.2
        # ),
        
        # GaussianBlur: 배경의 미세한 질감을 뭉개서 큰 구조에 집중하도록 함
        transforms.GaussianBlur(
            kernel_size=(3, 3), # 커널 크기 범위 증가
            sigma=(0.1, 3.0)    # 시그마(흐림 강도) 범위 증가
        ),

        # 텐서 변환 및 후속 증강
        # 이미지를 텐서로 변환 (주의: 이 시점부터 픽셀 값은 0~1)
        transforms.ToTensor(),
        
        # RandomErasing: 텐서에 적용. 이미지 일부를 가려 모델의 강건함을 높임
        transforms.RandomErasing(
            p=0.5,              # 확률을 다시 50%로 감소
            scale=(0.1, 0.2),  # 삭제 면적을 5% ~ 20%로 감소
            ratio=(0.3, 3.3),
            value=0
        ),
        
        # 정규화
        # 모델에 입력하기 직전, 표준 정규화 수행
        transforms.Normalize(mean=clip_mean, std=clip_std),
    ])

    # 검증(Validation) 및 테스트(Test)용 변환
    # 데이터 증강 없이, 리사이즈와 정규화만 수행
    val_transform = transforms.Compose([
        transforms.Resize(size=(224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std),
    ])
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
        sensor_transform=sensor_preprocessor, # 센서 전처리 적용
        threshold_epoch=THRESHOLD_EPOCH
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    if JSON_VAL_PATH:
        val_dataset = VideoSensorDataset(
            json_path=JSON_VAL_PATH,
            data_root=DATA_ROOT,
            num_frames=NUM_FRAMES,
            transform=val_transform,
            sensor_transform=sensor_preprocessor # 검증셋에도 동일한 전처리 적용
        )

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        
    else:
        val_loader = None

    print(f"Train dataset size: {len(train_dataset)}")
    
    if val_loader:
        print(f"Validation dataset size: {len(val_dataset)}")
    # ==================================================================


    # ==================================================================
    # 5. 모델, 손실 함수, 옵티마이저, 필요 파라미터 정의
    # ==================================================================

    # 첫 배치에서 데이터를 가져와 프로토타입 초기화
    # print("Initializing prototypes from first batch...")
    # prototypes = initialize_prototypes(
    #     train_dataloader, 
    #     num_clusters, 
    #     embedding_dim, 
    #     num_sensors, 
    #     device,
    #     args
    # )
    
    # 모델 초기화 (초기화된 프로토타입 전달)
    prototypes = initialize_prototypes(
        train_loader, 
        NUM_CLASSES, 
        EMBEDDING_DIM, 
        NUM_SENSORS, 
        DEVICE, 
        args
    )
    clustering_model = HybridClusteringModule(
        embedding_dim=EMBEDDING_DIM,
        num_sensors=NUM_SENSORS, 
        num_clusters=NUM_CLASSES,
        # val_dataloader=val_loader,
        prototypes=prototypes,
        alpha_fixed=ALPHA_FIXED,
        total_epochs=EPOCHS,
        threshold_epoch=THRESHOLD_EPOCH
    ).to(DEVICE)

    video_model = ViTWithCAM(num_classes=NUM_CLASSES).to(DEVICE)
    # sensor_model = MW2StackRNNPooling().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    parameters = list(video_model.parameters()) + list(clustering_model.parameters())
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
    best_val_loss = float('inf') # 최고 성능 저장을 위한 변수, 무한대로 초기화

    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(epoch)
        clustering_model.update_epoch(epoch)
        # --- 1. 학습 단계 ---
        train_loss, train_acc = train_one_epoch_with_cam(video_model=video_model, clustering_model=clustering_model, dataloader=train_loader, criterion=criterion, optimizer=optimizer, device=DEVICE, epoch=epoch, output_dir=output_dir, args=args)
        print(f"[Train] Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    #     # --- 2. 검증 단계 ---
        validation_one_epoch_with_cam(video_model=video_model, clustering_model=clustering_model, dataloader=val_loader, criterion=criterion, device=DEVICE, epoch=epoch, output_dir=output_dir, args=args)

    #     # val_loader가 정의되었을 경우에만 실행
    #     if val_loader:
    #         # val_loss, val_acc = validate(model, val_loader, criterion, DEVICE, epoch, PATCH_SIZE)
    #         val_loss, val_acc = validate(model, val_loader, criterion, DEVICE, epoch)
    #         print(f"  [Val]   Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    #         current_loss = val_loss # 모델 저장의 기준은 검증 손실
    #     else:
    #         # 검증 로더가 없으면 학습 손실을 기준으로 모델 저장
    #         current_loss = train_loss

    #     # --- 3. 최고 성능 모델 저장 ---
    #     # 현재 검증 손실이 이전에 기록된 최고 성능(최저 손실)보다 낮으면 모델을 저장
    #     if current_loss < best_val_loss:
    #         best_val_loss = current_loss
    #         torch.save(model.state_dict(), 'best_model.pth')
    #         print(f"  >> Best model saved with validation loss: {best_val_loss:.4f}")

    # print("\n--- Training Complete ---")
    # print(f"Final best validation loss: {best_val_loss:.4f}")
    # ==================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pure PyTorch version of SwAV for IMU Clustering")
    
    # 기존 인자들을 그대로 사용
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=9) # 더미데이터는 10개 클래스
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--sk_iterations", type=int, default=3)
    parser.add_argument("--project_name", type=str, default="SwAV_IMU_Clustering_PyTorch")
    parser.add_argument("--run_name", type=str, default="swav_hybrid_pytorch")
    parser.add_argument('--config', type=str, default='path/to/your/config.yaml') # 실제 경로로 수정 필요
    parser.add_argument("--alpha_fixed", type=bool, default=False) # --alpha_fixed 사용시 True
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--num_sensors", type=int, default=97)
    parser.add_argument("--threshold_epoch", type=int, default=15)
    
    args = parser.parse_args()
    main(args)