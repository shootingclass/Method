import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from tqdm import tqdm
import numpy as np
from .model import VideoTeacherMLP
from .module import COMODOLightningModule
import time

def initialize_comodo(args, datamodule):
    """
    COMODO 모델을 초기화하기 전에 필요한 모든 데이터(큐 등)를 준비합니다.
    DataModule 자체는 수정하지 않습니다.
    """
    print("--- [COMODO] Preparing special data (instance queue)... ---")
    # --- 0. 큐 캐시 파일 경로 정의 ---
    queue_cache_filename = f"instance_queue_s{args.seed}_r{args.queue_size_ratio}.pt"
    queue_cache_path = os.path.join(args.baseline_video_cache_dir, queue_cache_filename)
    datamodule.setup(stage='fit')
    full_train_dataset = datamodule.train_dataset
    
    # --- 1. 큐 캐시 파일이 이미 존재하는지 확인 ---
    if os.path.exists(queue_cache_path):
        print(f"--- Loading pre-computed instance queue from: {queue_cache_path} ---")
        try:
            cache = torch.load(queue_cache_path)
            instance_queue_encoded = cache["instance_queue"]
            train_indices = cache["train_indices"]
            print(f"--- Instance queue loaded successfully (Size: {len(instance_queue_encoded)}) ---")
            # 여기서 바로 큐 텐서를 반환하거나 comodo 모델에 설정하고 반환
            # model = COMODOLightningModule(...)
            # model.instance_queue = instance_queue_encoded.to(device) # GPU로 이동
            # return model
        except Exception as e:
            print(f"Warning: Failed to load queue cache {queue_cache_path}. Re-generating. Error: {e}")
    else:
        # 1. 큐에 포함될 인덱스 미리 결정
        start_time = time.time() # 시간 측정 시작
        # 1. 데이터 준비를 위해 DataModule의 훅을 '수동'으로 호출합니다.
        datamodule.prepare_data()
        datamodule.setup(stage='fit')

        # 2. 비디오 인코딩 캐시 생성 (On-demand Caching)
        #    전체 학습 데이터셋을 순회하며 캐시가 없는 파일만 인코딩합니다.
        device = 'cuda'

        os.makedirs(args.baseline_video_cache_dir, exist_ok=True) # exist_ok=True prevents error if dir already exists
        print("Checking and creating video feature cache for each sample...")
        video_teacher = VideoTeacherMLP(
            args.video_ckpt, args.mlp_output_dim, args.mlp_hidden_dim, device
        ).to(device)
        video_teacher.eval()
        indices = range(len(full_train_dataset))
        queue_size = int(len(full_train_dataset) * args.queue_size_ratio)
        idxs_in_queue = set(np.random.RandomState(args.seed).choice(
            indices, queue_size, replace=False,
        ))
        
        print(f"Assembling instance queue ({len(idxs_in_queue)} samples) and caching features...")
        
        queue_encoded_list = [None] * queue_size # 미리 리스트 공간 확보 (선택사항)
        queue_idx_map = {idx: i for i, idx in enumerate(idxs_in_queue)} # 원래 idx -> 큐 리스트 인덱스 매핑
        

        # 2. 데이터셋 순회 (한 번만!)
        with torch.no_grad():
            # tqdm으로 전체 데이터셋 순회
            for idx, batch in enumerate(tqdm(full_train_dataset, desc="[COMODO] Processing & Caching")):
                
                # 3. 현재 샘플이 큐에 필요한지 확인
                if idx in idxs_in_queue:
                    # __getitem__ 반환 순서 및 video_id 추출 방식 확인 필요
                    frames_tensor, _, _, item_ids, _ = batch 
                    try:
                        original_idx, video_id = item_ids # (idx, video_id) 튜플 가정
                    except ValueError:
                        video_id = item_ids # video_id 문자열 자체 가정
                    
                    cache_path = os.path.join(args.baseline_video_cache_dir, f"{video_id}.pt")
                    cache_dir = os.path.dirname(cache_path)
                    os.makedirs(cache_dir, exist_ok=True) # 캐시 디렉토리 생성

                    encoded_feature = None

                    # 4. 캐시 확인 및 로드
                    if os.path.exists(cache_path):
                        try:
                            encoded_feature = torch.load(cache_path, map_location='cpu') # CPU로 로드
                        except Exception as e:
                            print(f"Warning: Failed to load cache {cache_path}. Re-encoding. Error: {e}")
                            # 캐시 로드 실패 시 아래에서 재생성하도록 encoded_feature는 None 유지
                    
                    # 5. 캐시 없거나 로드 실패 시 -> 인코딩 및 저장
                    if encoded_feature is None:
                        if not isinstance(frames_tensor, torch.Tensor):
                            print(f"Warning: Expected tensor at index {idx}, got {type(frames_tensor)}. Skipping.")
                            continue 

                        # video_teacher로 인코딩 (GPU 사용)
                        encoded_video = video_teacher.encode(frames_tensor.unsqueeze(0).to(device))
                        
                        # CPU로 옮겨서 저장하고, 큐에도 추가 (중복 로드 방지)
                        encoded_feature = encoded_video.squeeze(0).cpu() 
                        torch.save(encoded_feature, cache_path)
                        print(f"Cached feature for {video_id}")

                    # 6. 큐 리스트에 추가 (미리 계산된 인덱스 사용)
                    list_index = queue_idx_map[idx]
                    queue_encoded_list[list_index] = encoded_feature
                    

        # None이 있는지 확인 (캐싱/로딩 실패 여부)
        final_queue = [item for item in queue_encoded_list if item is not None]
        if len(final_queue) != queue_size:
            raise RuntimeError(f"Warning: Final queue size ({len(final_queue)}) differs from target ({queue_size}). Some items failed.")
        try:
            print(f"--- Saving computed instance queue to: {queue_cache_path} ---")
            queue_cache_dir = os.path.dirname(queue_cache_path)
            os.makedirs(queue_cache_dir, exist_ok=True) # 큐 캐시 디렉토리 생성
            torch.save(final_queue, queue_cache_path)
            print(f"--- Instance queue saved successfully (Size: {final_queue.shape}) ---")
        except Exception as e:
            print(f"Error: Failed to save queue cache {queue_cache_path}. Error: {e}")
            
        end_time = time.time()
        print(f"Queue generation took {end_time - start_time:.2f} seconds.")

        del video_teacher # 메모리 확보

        instance_queue_encoded = torch.stack(final_queue)

        # 4. 실제 학습에 사용할 데이터셋을 'Subset'으로 재구성합니다.
        #    큐에 사용된 인덱스는 학습에서 제외합니다.
        train_indices = [i for i in indices if i not in idxs_in_queue]
                # --- save both queue and filtered training indices ---
        save_dict = {
            "instance_queue": instance_queue_encoded,
            "train_indices": train_indices,
        }
        torch.save(save_dict, queue_cache_path)
        print(f"Saved instance queue and filtered train dataset (train size={len(train_indices)})")

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