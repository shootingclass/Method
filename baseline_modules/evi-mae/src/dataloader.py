# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader_old.py

# modified from:
# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch

import csv
import json
import os.path

import pandas as pd
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import torchvision.transforms as T
from PIL import Image
import PIL
import decord
from transforms import GroupNormalize, GroupMultiScaleCrop, Stack, ToTorchFormatTensor

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1

    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        return mask 

class DataAugmentationForVideoMAE(object):
    def __init__(self, args=None, video_masking_ratio=0.9):
        self.input_size = 224 # 모든 프레임을 224x224로 처리!!
        self.mask_type = 'tube'
        self.window_size = (8,14,14) # 마스킹 단위 (Time, Height, Width)를 각각 (8, 14, 14)로 나눔
        self.mask_ratio = video_masking_ratio

        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std) # 모든 프레임에 동일하게 정규화 적용
        self.train_augmentation = GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]) # 다양한 비율로 프레임을 랜덤 크롭해서 Data Augmentation -> 결국 사이즈가 224x224로 조정된다!!
        
        self.transform = T.Compose([                            
            self.train_augmentation,
            Stack(roll=False), # 프레임들을 하나의 텐서로 쌓음 (T, C, H, W 등 형태로) -> 16프레임 비디오라면 이는 결국 16장의 이미지고, 비디오는 이미지(프레임)의 연속체 이므로 시간 순서대로 쌓는다!!
            ToTorchFormatTensor(div=True), # float32로 변환 및 정규화 범위 맞춤
            normalize,
        ])
        
        # 시간-공간적으로 연속된 마스킹 위치를 랜덤하게 생성
        if self.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                self.window_size, self.mask_ratio
            )
    
    # __call__ 함수는 DataAugmentationForVideoMAE 인스턴스를 함수처럼 쓸 때 호출됨
    def __call__(self, images):
        
        # 여기서 images는 프레임!!
        # 프레임 전처리 수행
        process_data, _ = self.transform(images)
        
        # 마스킹 위치 생성
        # 결국 process_data는 (T*C, H, W) 형태 텐서, mask는 마스킹 위치
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


# EVIDataset(args.data_train, imu_conf=imu_conf, label_csv=args.label_csv, video_masking_ratio=args.video_masking_ratio, image_as_video=args.image_as_video)로 호출
class EVIDataset(Dataset):
    def __init__(self, dataset_json_file, imu_conf, label_csv=None, video_masking_ratio=0.9, image_as_video=False):
        """
        Dataset that manages imu recordings
        :param imu_conf: Dictionary containing the imu loading and preprocessing settings
        :param dataset_json_file
        """

        self.use_imu = True
        self.datapath = dataset_json_file
        
        if 'wear' in self.datapath:
            self.dataset_name = 'wear'
            
        elif 'cmu' in self.datapath:
            self.dataset_name = 'cmu'
            
        else:
            self.dataset_name = 'opp'
        
        # /home/junho/IMU-Video-MAE/data_release/Opportunity++
        # self.data_base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(self.datapath))))
        self.data_base_path = os.path.dirname(os.path.dirname(os.path.dirname(self.datapath)))
        
        # JSON 파일 열어서 데이터 로드
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        # JSON 데이터 전처리
        self.data = data_json['data']
        self.data = self.pro_data(self.data)
        print('Dataset has {:d} samples'.format(self.data.shape[0]))
        self.num_samples = self.data.shape[0]

        # IMU 설정 로드
        self.imu_conf = imu_conf
        self.label_smooth = self.imu_conf.get('label_smooth', 0.0)
        print('Using Label Smoothing: ' + str(self.label_smooth))
        
        self.melbins = self.imu_conf.get('num_mel_bins')
        self.freqm = self.imu_conf.get('freqm', 0)
        self.timem = self.imu_conf.get('timem', 0)
        print('now using following mask: {:d} freq, {:d} time'.format(self.imu_conf.get('freqm'), self.imu_conf.get('timem')))
        
        self.mixup = self.imu_conf.get('mixup', 0)
        print('now using mix-up with rate {:f}'.format(self.mixup))
        
        self.dataset = self.imu_conf.get('dataset')
        print('now process ' + self.dataset)

        # 데이터 정규화 설정
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.imu_conf.get('mean')
        self.norm_std = self.imu_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.imu_conf.get('skip_norm') if self.imu_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        
        # 데이터 증강을 위한 노이즈 추가 설정
        # imu_conf에 noise 키가 있으면 그 값을 사용, noise 키가 없으면 False를 기본값으로 사용
        # if add noise for data augmentation
        self.noise = self.imu_conf.get('noise', False)
        if self.noise == True:
            print('now use noise augmentation')
        else:
            print('not use noise augmentation')

        # label 처리
        # self.index_dict -> {'404505': '0', '404508': '1', '404511': '2', '404516': '3', '404517': '4', '404519': '5', '404520': '6', '405506': '7', '406505': '8', '406508': '9', '406511': '10', '406516': '11', '406517': '12', '406519': '13', '406520': '14', '407521': '15', '408512': '16'}
        self.index_dict = make_index_dict(label_csv)
        
        self.label_num = len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))

        # 타겟 길이 설정
        self.target_length = self.imu_conf.get('target_length')

        # 모드 설정
        # train or eval
        self.mode = self.imu_conf.get('mode')
        print('now in {:s} mode.'.format(self.mode))

        # no use
        # set the frame to use in the eval mode, default value for training is -1 which means random frame
        self.frame_use = self.imu_conf.get('frame_use', -1)
        # by default, 10 frames are used
        self.total_frame = self.imu_conf.get('total_frame', 10)
        # print('now use frame {:d} from total {:d} frames'.format(self.frame_use, self.total_frame))


        # 이미지 전처리 설정(왜 있는지 모르겠음)
        # by default, all models use 224*224, other resolutions are not tested
        self.im_res = self.imu_conf.get('im_res', 224) # 224
        print('now using {:d} * {:d} image input'.format(self.im_res, self.im_res))
        
        self.preprocess = T.Compose([
            T.Resize(self.im_res, interpolation=PIL.Image.BICUBIC),
            T.CenterCrop(self.im_res),
            T.ToTensor(),
            T.Normalize(
                mean=[0.4850, 0.4560, 0.4060],
                std=[0.2290, 0.2240, 0.2250]
            )])

        
        # 스펙트로그램 생성
        # 아래의 코드 시나리오를 설명하자면...
        # 오디오 신호는 샘플들의 연속이고, 그 중 일부를 윈도우로 잘라서 분석함
        # 하나의 윈도우는 24개의 연속된 샘플로 구성됨 (win_length=24)
        # 샘플 하나씩 이동하며 다음 윈도우를 만듦 (hop_length=1) → 높은 시간 해상도
        # 각 윈도우마다, 24개의 실제 데이터에 대해 256 포인트 FFT를 수행 (n_fft=256) → 부족한 부분(256 - 24)은 제로 패딩(zero-padding) 해서 FFT 계산 가능하게 함
        
        # 예시
        # win_length -> [x₀, x₁, x₂, ..., x₂₃], 총 24개 샘플
        # n_fft -> [x₀, x₁, ..., x₂₃, 0, 0, ..., 0] 처럼 뒤에 0을 232개 추가해서 총 256개 만들고, 이걸 FFT에 넣는거임
        # self.spectrogram_transform = torchaudio.transforms.Spectrogram(
        #     n_fft=256, # 한번의 FFT 연산에 사용할 포인트가 256개
        #     win_length=24, # 한 윈도우가 24개의 샘플로 구성됨
        #     hop_length=1, # 윈도우가 매번 1샘플씩 이동
        #     window_fn=torch.hann_window # Hann 윈도우를 적용하여 변환 수행
        # )

        self.spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=256, 
            win_length=24, # 한 윈도우가 24개의 샘플로 구성됨
            hop_length=1, # 윈도우가 매번 1샘플씩 이동
            window_fn=torch.hann_window # Hann 윈도우를 적용하여 변환 수행
        )

        # 비디오 데이터 증강 설정
        # for videomae
        self.video_transform = DataAugmentationForVideoMAE(video_masking_ratio=video_masking_ratio)
        self.num_segments = 1
        self.skip_length = 64
        self.new_step = 4
        self.new_length = 16
        assert self.skip_length / self.new_step == self.new_length
        self.temporal_jitter = False

        # 시각화 및 이미지-비디오 설정
        self.save_visualization_path = None # './visualization_dataloader_wear/' # if None, no visualization
        self.image_as_video = image_as_video


    # change python list to numpy array to avoid memory leak.
    def pro_data(self, data_json):
        for i in range(len(data_json)):
            data_json[i] = [data_json[i]['imu_path'], data_json[i]['label'], data_json[i]['video_id'], data_json[i]['frame_path']]
        
        # dtype=str**로 넘기니까 원래 int였던 'label'도 문자형으로 바뀌게 됨
        data_np = np.array(data_json, dtype=str)
        return data_np


    def decode_data(self, np_data):
        datum = {}
        if self.use_imu:
            datum['imu'] = np_data[0]
            datum['labels'] = np_data[1]
            datum['video_id'] = np_data[2]
            datum['video_path'] = np_data[3]
        return datum
    

    def _imu2fbank(self, filename, video_frame_id_list, video_duration, filename2=None, mix_lambda=-1):
        
        # print(f"video_frame_id_list: {video_frame_id_list}")
        # print(f"video_duration: {video_duration}")
        
        fbank_list = []
        raw_list = []
        
        if self.dataset_name == 'cmu':
            imu_to_use = [1,8,15,  5,12,19,  3,10,17,  4,11,18] # xyz acceleration for left arm, right arm, left leg, right leg
        
        elif self.dataset_name == 'wear':
            imu_to_use = [10,11,12, 1,2,3, 7,8,9, 4,5,6] # xyz acceleration for left arm, right arm, left leg, right leg

        elif self.dataset_name == 'opp':
            imu_to_use = [50,51,52, 76,77,78, 102,103,104, 118,119,120] # change to sensor columns

        # 데이터베이스 경로와 파일 이름을 결합하여 전체 파일 경로를 생성
        filename = os.path.join(self.data_base_path, filename)
        
        # CSV 파일에서 IMU 데이터를 읽어 NumPy 배열로 변환
        IMU_data = pd.read_csv(filename, index_col=False).to_numpy() # 250, 14 for wear; 150, 64 for cmu
        
        # 논문에서 말하던 Cleaning!!
        # 비디오의 특정 프레임 ID와 비디오 지속 시간을 사용하여 IMU 데이터의 시작과 끝 인덱스를 계산
        # 비디오 전체를 사용하는게 아니라, 비디오의 특정 프레임들을 샘플링 했기 때문에, IMU의 지속 시간도 같이 조정해 줘야 함
        IMU_start = int(video_frame_id_list[0] / video_duration * IMU_data.shape[0])
        IMU_end = int(video_frame_id_list[-1] / video_duration * IMU_data.shape[0])
        
        # 계산된 인덱스를 사용하여 IMU 데이터를 자름
        IMU_data = IMU_data[IMU_start:IMU_end, :] # [~64, 64]

        # 각 IMU 채널에 대해 스펙트로그램 생성 및 전처리
        for imu_idx in imu_to_use:
            
            # no mixup
            # 선택된 IMU 채널의 데이터를 추출하고, 평균을 제거하여 정규화
            # 원본 데이터는 raw_list에 저장
            if filename2 == None:
                
                # CSV에서 읽어온 IMU_data에서 필요한 채널 인덱스(imu_idx)에 해당하는 열만 추출하여, one_IMU_wave 변수에 저장
                one_IMU_wave = IMU_data[:, imu_idx]
                
                # float32 타입으로 변환
                one_IMU_wave = one_IMU_wave.astype(np.float32)
                
                # 원본 데이터 저장
                raw_list.append(one_IMU_wave)
                
                # NumPy 배열을 PyTorch 텐서로 변환
                # one_IMU_wave[:,None] -> 기존 [데이터 길이] 형태의 1차원 배열을 [데이터 길이, 1] 형태의 2차원 배열로 바꿔줌
                # transpose(0, 1) -> [데이터 길이, 1] 형태를 [1, 데이터 길이] 형태로 바꿈
                # 오디오 처리나 스펙트로그램 변환 시 일반적으로 [채널 수, 샘플 수] 형태를 많이 쓰는데, 이를 맞추기 위한 조치
                waveform = torch.from_numpy(one_IMU_wave[:,None]).transpose(0,1)

                # 논문에서 말하던 Normalization!!
                # 평균값을 빼서 시그널의 평균을 0으로 맞추는 정규화 과정
                waveform = waveform - waveform.mean()
                
            # mixup
            else:
                waveform1, sr = torchaudio.load(filename)
                waveform2, _ = torchaudio.load(filename2)

                # mixup 시 IMU 데이터 정규화
                waveform1 = waveform1 - waveform1.mean()
                waveform2 = waveform2 - waveform2.mean()

                if waveform1.shape[1] != waveform2.shape[1]:
                    if waveform1.shape[1] > waveform2.shape[1]:
                        # padding
                        temp_wav = torch.zeros(1, waveform1.shape[1])
                        temp_wav[0, 0:waveform2.shape[1]] = waveform2
                        waveform2 = temp_wav
                    else:
                        # cutting
                        waveform2 = waveform2[0, 0:waveform1.shape[1]]

                mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
                waveform = mix_waveform - mix_waveform.mean()

            
            # 논문에서 말하던 Resampling!!
            try:

                resample_ratio = 64/waveform.shape[1] 
                resample_target = int(100*resample_ratio)              
                waveform = torchaudio.transforms.Resample(30, resample_target)(waveform)
                spectrogram = self.spectrogram_transform(waveform)                 
                spectrogram_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)
                
                # squeeze(0) -> (채널=1, 주파수, 시간) 같은 텐서에서 채널 차원이 1이면 해당 차원을 제거, 예: [1, 129, 321] → [129, 321]
                # transpose(0,1) -> 차원 0(주파수)과 차원 1(시간)을 바꿈, 예: [129, 321] → [321, 129]
                # [:,:128] : 129개의 주파수 축 중 앞의 128개만 선택, 예: [321,129] -> [321,128]
                # 128개만 선택하는건, 논문에서 For the IMU signals, we transform them into spectrograms with a temporal dimension of Timu = 160 and a frequency dimension of Mimu = 128 라고 명시했기 때문!!
                fbank = spectrogram_db.squeeze(0).transpose(0,1)[:,:128] 
                
            except:
                fbank = torch.zeros([512, 128]) + 0.01
                print('there is a loading error')
                exit()
            
            # "항상 320 프레임 길이를 갖도록 만들겠다”와 같은 설정을 했다면, target_length는 320일 수 있음
            target_length = self.target_length
            
            # fbank는 (시간, 주파수) shape이므로, shape[0]은 시간(프레임 수)
            n_frames = fbank.shape[0]
            
            # 목표 길이와 현재 길이의 차이를 계산
            # 위에서 봤듯이, 스펙트로그램의 프레임이 321 프레임이라 조금의 차이가 있음
            p = target_length - n_frames

            # 현재 길이가 부족하면 padding
            if p > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                fbank = m(fbank)
            
            # 현재 길이가 목표 길이보다 길면 잘라냄
            elif p < 0:
                fbank = fbank[0:target_length, :]

            # 원래 [시간, 주파수] 형태였던 텐서를 [1, 시간, 주파수] 형태로 만들어줌
            fbank = fbank.unsqueeze(0)
            fbank_list.append(fbank)
        
        # 각 IMU 채널별로 생성된 필터뱅크 스펙트로그램을 하나의 텐서로 결합
        # 각 텐서가 [1, time, freq] 형태일 때, 리스트에 12개가 있으면 최종적으로 [12, time, freq] 형태로 합쳐짐
        # 나중에 모델의 forward 함수에 들어가면 다시 부위별로 3개씩 나눠주는 과정이 있음
        fbank_cat = torch.cat(fbank_list, dim=0) 
        
        print('fbank 모양', fbank_cat.shape)
        
        # 원본 IMU 신호를 NumPy 배열로 결합
        raw_cat = np.array(raw_list) # [12, 60]
        
        # fbank는 전처리된 스펙트로그램, raw는 원본 IMU 데이터        
        return (fbank_cat, raw_cat)


    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length + 1) // self.num_segments # skip_length = 32, num_segments = 1
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),average_duration)
            offsets = offsets + np.random.randint(average_duration,size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(np.random.randint(
                num_frames - self.skip_length + 1,
                size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter: # False
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
            
        return offsets + 1, skip_offsets


    def _video_TSN_decord_batch_loader(self, directory, video_reader, duration, indices, skip_offsets):
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)): # skip_length = 32, new_step = 2
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step

        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory, duration))
        return sampled_list, frame_id_list, duration


    def get_video(self, video_name):
        video_name = os.path.join(self.data_base_path, video_name)        
        decord_vr = decord.VideoReader(video_name, num_threads=1)
        
        # 비디오의 전체 프레임 수 계산
        duration = len(decord_vr)

        # 프레임을 어떻게 추출할지 결정
        segment_indices, skip_offsets = self._sample_train_indices(duration)
        
        # segment_indices에 기반해 선택된 프레임들만 로드
        images, frame_id_list, duration = self._video_TSN_decord_batch_loader(video_name, decord_vr, duration, segment_indices, skip_offsets)
        
        # self.save_visualization_path = None 이므로 필요 없음
        if self.save_visualization_path != None:
            if not os.path.exists(self.save_visualization_path):
                os.makedirs(self.save_visualization_path)
            # save images as a video with PIL
            save_video_name = '%s/%s.gif' % (self.save_visualization_path, video_name.split('/')[-1].split('.')[0])
            print('save video to', save_video_name)
            images[0].save(save_video_name, save_all=True, append_images=images[1:], duration=200, loop=0)
        
        # 비디오 전처리(가장 중요한 부분임)
        process_data, mask = self.video_transform((images, None)) # T*C,H,W
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W 형태로 모델 입력에 맞춤

        # image_as_video = False 이므로 필요 없음
        if self.image_as_video:
            chosen_image = process_data[:, 8, :, :]
            # fill process_data with chosen_image
            process_data = chosen_image.unsqueeze(1).repeat(1, 16, 1, 1)

        return process_data, mask, frame_id_list, duration


    # 1. 데이터 로딩
    # 2. 데이터 전처리
    # 3. 레이블 생성
    # 4. 데이터와 레이블 반환
    def __getitem__(self, index):

        if random.random() < self.mixup:
            print('should not reach here not implemented')
            exit()

        else:
            datum = self.data[index]

            # {'imu': 'trim_5s_IMU/sbj_17_1470_1475.csv', 'labels': '0', 'video_id': 'sbj_17_1470_1475', 'video_path': 'trim_5s_400x300_correct60/sbj_17_1470_1475.mp4'} -> wear dataset           
            # {'imu': np.str_('trim_5s_imu/S3-ADL3/S3-ADL3_000730329_000733329_imu.csv'), 'labels': np.str_('406511'), 'video_id': np.str_('S3-ADL3_000730329_000733329'), 'video_path': np.str_('trim_5s_video/S3-ADL3/S3-ADL3_000730329_000733329.mp4')} -> opp dataset
            datum = self.decode_data(datum)

            # process_data는 비디오 데이터를 모델 학습에 사용 가능한 텐서(C, T, H, W)로 정제한 결과
            # 정확하게는 [3, 16, 224, 224] -> 논문에서 비디오는 16프레임으로 다운샘플링 한다고 했음!!
            process_data, mask, video_frame_id_list, video_duration = self.get_video(datum['video_path'])

            try:                
                # fbank는 전처리된 spectrogram
                # raw는 원본 IMU 데이터
                fbank, raw_imu = self._imu2fbank(datum['imu'], video_frame_id_list, video_duration)
                fbank = fbank.to(torch.float32)
                
            except:
                fbank = torch.zeros([self.target_length, 6]) + 0.01
                print('there is an error in loading imu')
                exit()
                
            # label smoothing 적용 부분
            
            # soft label 벡터 초기화
            # 모든 클래스에 대해 (epsilon / 클래스 수) 값 부여 → 이게 label smoothing의 핵심
            # 클래스 수가 5, label_smooth = 0.1이면 → 모든 값은 0.02로 시작함
            label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num) 

            # datum['labels']는 정수 형태의 정답 클래스 (예: 3)
            # 해당 위치에 1 - epsilon 값을 넣음
            # 정답이 "2"이고 label_smooth = 0.1이면 → label_indices[2] = 0.9 → 나머지 값은 전부 0.02
            label_idx = int(datum['labels'])
            label_indices[label_idx] = 1.0 - self.label_smooth
    
            # 최종적으로 이 soft label을 PyTorch Tensor로 변환
            # 이 Tensor는 이후에 "classification loss" 계산 시 사용됨
            # label_indices는 tensor([0.0056, 0.0056, 0.0056, 0.0056, 0.0056, 0.0056, 0.0056, 0.0056, 0.9000, 0.0056, 0.0056, 0.0056, 0.0056, 0.0056, 0.0056, 0.0056, 0.0056, 0.0056]) 이런 식으로 생김
            label_indices = torch.FloatTensor(label_indices)

        # normalize the input for both training and test
        if self.skip_norm == False:
            fbank = (fbank - self.norm_mean) / (self.norm_std)
            
        # skip normalization the input ONLY when you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            if self.use_imu:
                fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1], fbank.shape[2]) * np.random.rand() / 10
                # fbank = torch.roll(fbank, np.random.randint(-self.target_length, self.target_length), 0)
            else:
                fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
                fbank = torch.roll(fbank, np.random.randint(-self.target_length, self.target_length), 0)
        
        # fbank는 전처리된 spectrogram
        # process_data는 비디오 데이터를 모델 학습에 사용 가능한 텐서(C, T, H, W)로 정제한 결과
        # datum['video_path']는 'trim_5s_400x300_correct60/sbj_17_1470_1475.mp4' 과 같이 그냥 경로임
        # label_indices는 레이블
        return fbank, process_data, datum['video_path'], label_indices

    def __len__(self):
        return self.num_samples