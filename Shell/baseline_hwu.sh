python main.py --dataset_name HWU-USP --model_name primus --batch_size 16
echo "primus 끝"
python main.py --dataset_name HWU-USP --model_name imu2clip --batch_size 16
echo "imu2clip 끝"
python main.py --dataset_name HWU-USP --model_name comodo --batch_size 8
echo "comodo 끝"
python main.py --dataset_name HWU-USP --model_name mae --batch_size 8
echo "mae 끝"