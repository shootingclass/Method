import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from tqdm import tqdm
import numpy as np
from .model import VideoTeacherMLP
from .module import COMODOLightningModule

def initialize_comodo(args, datamodule):
    """
    COMODO 모델을 초기화하기 전에 필요한 모든 데이터(큐 등)를 준비합니다.
    DataModule 자체는 수정하지 않습니다.
    """
    print("--- [COMODO] Preparing special data (instance queue)... ---")
    
    # 1. 데이터 준비를 위해 DataModule의 훅을 '수동'으로 호출합니다.
    datamodule.prepare_data()
    datamodule.setup(stage='fit')

    # 2. 비디오 인코딩 캐시 생성 (On-demand Caching)
    #    전체 학습 데이터셋을 순회하며 캐시가 없는 파일만 인코딩합니다.
    full_train_dataset = datamodule.train_dataset
    device = 'cuda' if args.devices > 0 else 'cpu'
    video_teacher = VideoTeacherMLP(
        args.video_ckpt, args.mlp_output_dim, args.mlp_hidden_dim
    ).to(device)
    video_teacher.eval()

    print("Checking and creating video feature cache for each sample...")
    with torch.no_grad():
        for idx in tqdm(range(len(full_train_dataset)), desc="[COMODO] Caching video features"):
            sample_info = full_train_dataset.samples[idx]
            frames_tensor, _, _, video_id= sample_info
            video_id = sample_info['video_id']
            cache_path = os.path.join(args.baseline_video_cache_dir, f"{video_id}.pt")

            if not os.path.exists(cache_path):
                # 캐시가 없으면, 데이터셋에서 원본 비디오를 가져와 인코딩
                encoded_video = video_teacher.encode(frames_tensor.unsqueeze(0).to(device))
                # CPU로 옮겨서 저장
                torch.save(encoded_video.squeeze(0).cpu(), cache_path)
    
    del video_teacher # 메모리 확보

    # 3. 인스턴스 큐 생성
    #    이제 모든 캐시가 준비되었으므로, 필요한 파일만 빠르게 로드합니다.
    indices = range(len(full_train_dataset))
    idxs_in_queue = set(np.random.RandomState(args.seed).choice(
        indices, args.queue_size, replace=False,
    ))
    
    print(f"Assembling instance queue with {len(idxs_in_queue)} samples from cache...")
    queue_encoded_list = []
    
    for idx in tqdm(idxs_in_queue, desc="[COMODO] Assembling instance queue"):
        video_id = full_train_dataset.samples[idx]['video_id']
        cache_path = os.path.join(args.baseline_video_cache_dir, f"{video_id}.pt")
        queue_encoded_list.append(torch.load(cache_path))

    instance_queue_encoded = torch.stack(queue_encoded_list)

    # 4. 실제 학습에 사용할 데이터셋을 'Subset'으로 재구성합니다.
    #    큐에 사용된 인덱스는 학습에서 제외합니다.
    train_indices = [i for i in indices if i not in idxs_in_queue]
    datamodule.train_dataset = Subset(full_train_dataset, train_indices)
    
    print(f"Original dataset size: {len(full_train_dataset)}")
    print(f"Instance queue size: {len(instance_queue_encoded)}")
    print(f"Final training dataset size: {len(datamodule.train_dataset)}")
    
    # 5. (선택사항) 앵커 임베딩 생성 - Pre-training에서는 불필요
    anchor_embeddings_tensor = None 

    print("--- [COMODO] Data preparation complete ---")
    
    # 6. 준비된 데이터로 COMODO LightningModule 인스턴스 생성
    model = COMODOLightningModule(args, instance_queue_encoded, anchor_embeddings_tensor)
    return model

class COMODOLoss(nn.Module):
    def __init__(
        self,
        instanceQ_encoded,
        student_model,
        teacher_temp=0.1,
        student_temp=0.05,
    ):
        """
        student_model:    IMU model
        teacher_model:    Video model
        teacher_temp:   distillation temperature for teacher model
        student_temp:   distillation temperature for student model
        """
        super(COMODOLoss, self).__init__()
        self.instanceQ_encoded = instanceQ_encoded
        self.student_model = student_model
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

    def forward(
        self,
        imu_features: torch.Tensor,
        z_v: torch.Tensor,
        input_mask: torch.Tensor = None,
    ):

        batch_size = z_v.shape[0]

        z_x = F.normalize(self.student_model(imu_features, input_mask), p=2, dim=1)

        # insert the current batch embedding from T
        instanceQ_encoded = self.instanceQ_encoded
        Q = torch.cat((instanceQ_encoded, z_v))

        # probability scores distribution for T, S: B X (N + 1)
        P_v = torch.einsum("nc,ck->nk", z_v, Q.t().clone().detach())
        P_x = torch.einsum("nc,ck->nk", z_x, Q.t().clone().detach())

        # FKL
        # Apply temperatures for soft-labels
        P_v = F.softmax(P_v / self.teacher_temp, dim=1)
        P_x = P_x / self.student_temp

        # loss computation, use log_softmax for stable computation
        loss = -torch.mul(P_v, F.log_softmax(P_x, dim=1)).sum() / batch_size

        # update the random sample queue
        self.instanceQ_encoded = Q[batch_size:]

        return loss