import numpy as np
import json
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2

# JSON에서 직접 비디오 경로 찾기
with open('/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window/action/linear_train.json', 'r') as f:
    raw_data = json.load(f)

data = raw_data['data']  # "data" 키 아래에 리스트가 있음

open_door1_paths = []
close_door1_paths = []

for item in data:
    if item['label'] == 0 and len(open_door1_paths) < 4:
        open_door1_paths.append(item['frame_path'])
    elif item['label'] == 2 and len(close_door1_paths) < 4:
        close_door1_paths.append(item['frame_path'])

print('Open Door 1:', open_door1_paths)
print('Close Door 1:', close_door1_paths)

base_path = '/mnt/hdd4tb/junho/Opportunity++/data_processed_2s_window'

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# mp4에서 중간 프레임 추출
def get_middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_idx = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

for i, frame_path in enumerate(open_door1_paths):
    full_path = os.path.join(base_path, frame_path)
    img = get_middle_frame(full_path)
    if img is not None:
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Open Door 1 - {i+1}', fontsize=12)
    axes[0, i].axis('off')

for i, frame_path in enumerate(close_door1_paths):
    full_path = os.path.join(base_path, frame_path)
    img = get_middle_frame(full_path)
    if img is not None:
        axes[1, i].imshow(img)
        axes[1, i].set_title(f'Close Door 1 - {i+1}', fontsize=12)
    axes[1, i].axis('off')

plt.suptitle('Open Door 1 vs Close Door 1: Middle Frames', fontsize=16)
plt.tight_layout()
plt.savefig('./embedding_viz_full/door1_raw_samples.png', dpi=150)
print('Saved!')

