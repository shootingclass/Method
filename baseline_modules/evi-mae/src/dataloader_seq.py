# -*- coding: utf-8 -*-
# Sequence-level dataloader for HWU dataset
# Each sample = one sequence = multiple 2s windows → stacked for mean pooling

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from dataloader import EVIDataset


class EVISequenceDataset(EVIDataset):
    """
    Sequence-level dataset for HWU.
    JSON 구조:
    {
        "data": [
            {
                "sequence_id": "tidy_s02",
                "windows": [
                    {"sensor_path": "...", "frame_path": "...", "class_name": "tidy", "label": 6, ...},
                    ...
                ]
            }
        ]
    }
    
    __getitem__ returns:
        sensor_stack: (S, C, H_spec, W_spec) — stacked fbank spectrograms
        video_stack:  (S, C_vid, T_vid, H_vid, W_vid) — stacked video tensors
        label:        (n_class,) — soft label (sequence-level, from class_name)
        meta:         dict with sequence_id and length
    """

    def __init__(self, dataset_json_file, imu_conf, label_csv=None,
                 video_masking_ratio=0.9, image_as_video=False,
                 class_name_to_idx=None):
        """
        class_name_to_idx: dict mapping class_name → int index.
                           If None, auto-build from label_csv.
        """
        # Store class_name_to_idx BEFORE super().__init__ tries to call pro_data
        self._class_name_to_idx = class_name_to_idx or {}
        self._sequences = []  # will be populated by pro_data (called in __init__)
        
        # Call parent __init__ — this will call our overridden pro_data
        super().__init__(dataset_json_file, imu_conf, label_csv,
                         video_masking_ratio, image_as_video)
        self.label_num = 5

    def pro_data(self, data_json):
        """
        Override: data_json here is the raw JSON list.
        For sequence format, each item has "sequence_id" and "windows".
        For flat format (backward compat), each item has "sensor_path" etc.
        """
        # Check if sequence format
        if len(data_json) > 0 and "sequence_id" in data_json[0]:
            return self._pro_data_sequence(data_json)
        else:
            # Flat format — fall back to parent
            return super().pro_data(data_json)

    def _pro_data_sequence(self, data_json):
        """Process sequence-format JSON data."""
        self._sequences = []
        
        for seq in data_json:
            seq_id = seq["sequence_id"]
            windows = seq["windows"]
            if len(windows) == 0:
                continue
            
            # Sequence-level label from class_name
            class_name = windows[0]["class_name"]
            
            # Auto-build class_name_to_idx if needed
            if class_name not in self._class_name_to_idx:
                self._class_name_to_idx[class_name] = len(self._class_name_to_idx)
            
            class_idx = self._class_name_to_idx[class_name]
            
            # Collect per-window data paths
            window_data = []
            for w in windows:
                window_data.append({
                    'sensor_path': w['sensor_path'],
                    'frame_path': w['frame_path'],
                    'label': class_idx,  # sequence-level class idx
                    'video_id': w.get('video_id', ''),
                })
            
            self._sequences.append({
                'sequence_id': seq_id,
                'class_name': class_name,
                'class_idx': class_idx,
                'windows': window_data,
            })
        
        # Return a dummy np array for compatibility (parent stores self.data and self.num_samples)
        # We override __len__ and __getitem__ so this is just for the print statement
        dummy = np.array([['' for _ in range(4)] for _ in range(len(self._sequences))], dtype=str)
        return dummy

    def __len__(self):
        if len(self._sequences) > 0:
            return len(self._sequences)
        return self.num_samples

    def __getitem__(self, index):
        seq = self._sequences[index]
        
        fbank_list = []
        video_list = []
        
        for w in seq['windows']:
            try:
                # Load video (same as parent)
                process_data, mask, video_frame_id_list, video_duration = self.get_video(w['frame_path'])
                
                # Load sensor fbank (same as parent)
                fbank, raw_imu = self._imu2fbank(w['sensor_path'], video_frame_id_list, video_duration)
                fbank = fbank.to(torch.float32)
                
                # Normalize
                if self.skip_norm == False:
                    fbank = (fbank - self.norm_mean) / (self.norm_std)
                
                # Noise augmentation
                if self.noise == True:
                    if self.use_imu:
                        fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1], fbank.shape[2]) * np.random.rand() / 10
                
                fbank_list.append(fbank)
                video_list.append(process_data)
                
            except Exception as e:
                print(f'[WARN] Error loading window {w["video_id"]} in seq {seq["sequence_id"]}: {e}')
                continue
        
        # If all windows failed, create dummy
        if len(fbank_list) == 0:
            print(f'[WARN] All windows failed for seq {seq["sequence_id"]}, creating dummy')
            fbank_list.append(torch.zeros([self.imu_conf.get('imu_channel_num', 6),
                                           self.target_length, 128]) + 0.01)
            video_list.append(torch.zeros([3, 16, 224, 224]))
        
        # Stack windows → (S, C, H, W) for sensor, (S, C, T, H, W) for video
        sensor_stack = torch.stack(fbank_list, dim=0)  # (S, C_imu, T_spec, F_spec)
        video_stack = torch.stack(video_list, dim=0)    # (S, C_vid, T_vid, H_vid, W_vid)
        
        # Label (soft label with label smoothing)
        label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num)
        label_indices[seq['class_idx']] = 1.0 - self.label_smooth
        label_indices = torch.FloatTensor(label_indices)
        
        meta = {
            'sequence_id': seq['sequence_id'],
            'class_name': seq['class_name'],
            'length': len(fbank_list),
        }
        
        return sensor_stack, video_stack, label_indices, meta


def collate_sequence(batch):
    """
    Collate variable-length sequences with padding.
    
    Input batch: list of (sensor_stack, video_stack, label, meta)
    Output:
        sensor_padded: (B, S_max, C, H, W)
        video_padded:  (B, S_max, C, T, H, W)
        lengths:       (B,)
        labels:        (B, n_class)
        metas:         list of dict
    """
    sensor_seqs, video_seqs, labels, metas = zip(*batch)
    
    B = len(sensor_seqs)
    lengths = torch.tensor([s.shape[0] for s in sensor_seqs], dtype=torch.long)
    S_max = int(max(lengths).item())
    
    # Sensor shape: each is (S_i, C, H, W)
    C_s = sensor_seqs[0].shape[1]
    H_s = sensor_seqs[0].shape[2]
    W_s = sensor_seqs[0].shape[3]
    
    sensor_padded = torch.zeros((B, S_max, C_s, H_s, W_s), dtype=sensor_seqs[0].dtype)
    for i, s in enumerate(sensor_seqs):
        sensor_padded[i, :s.shape[0]] = s
    
    # Video shape: each is (S_i, C, T, H, W)
    C_v = video_seqs[0].shape[1]
    T_v = video_seqs[0].shape[2]
    H_v = video_seqs[0].shape[3]
    W_v = video_seqs[0].shape[4]
    
    video_padded = torch.zeros((B, S_max, C_v, T_v, H_v, W_v), dtype=video_seqs[0].dtype)
    for i, v in enumerate(video_seqs):
        video_padded[i, :v.shape[0]] = v
    
    labels = torch.stack(labels, dim=0)
    
    return sensor_padded, video_padded, lengths, labels, list(metas)
