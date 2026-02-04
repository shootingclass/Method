import os
import torch
import pytorch_lightning as pl
import torchmetrics
import torch
import random
####################################################################
from baseline_modules import BasePretrainModule
from .model import MW2StackRNNPoolingMultihead, Clip4CLIPModel
from .utils import *
from baseline_modules.loss import InfoNCE

class FeatureQueue():
    def __init__(self, size, dim, device):
        self.size = size
        self.dim = dim
        self.queue = torch.zeros((size, dim), dtype=torch.float32).cuda()
        self.ptr = 0
        self.device = device
    
    def enqueue(self, tensors):
        """
        Enqueue a batch of tensors.

        Args:
            tensors (torch.Tensor): A batch of tensors of shape (batch_size, dim).
        """
        batch_size = tensors.size(0)
        if batch_size > self.size:
            raise ValueError("Batch size cannot be larger than queue size.")
        
        end_ptr = (self.ptr + batch_size) % self.size

        if end_ptr > self.ptr:
            self.queue[self.ptr:end_ptr] = tensors
        else:
            part1_len = self.size - self.ptr
            self.queue[self.ptr:] = tensors[:part1_len]
            self.queue[:end_ptr] = tensors[part1_len:]
        
        self.ptr = end_ptr
    
    def to(self, device):
        self.queue = self.queue.to(device)
        self.device = device

    def dequeue_and_enqueue(self, tensors):
        """
        Dequeue and enqueue a batch of tensors.

        Args:
            tensors (torch.Tensor): A batch of tensors of shape (batch_size, dim).

        Returns:
            torch.Tensor: The updated queue.
        """
        self.enqueue(tensors)
        return self.queue
    
    def get_queue(self):
        return self.queue
    
    def find_nearest_neighbors(self, tensors, k):
        """
        Find the k nearest neighbors in the queue for each tensor in the batch.

        Args:
            tensors (torch.Tensor): A batch of tensors of shape (batch_size, dim).
            k (int): The number of nearest neighbors to find.

        Returns:
            torch.Tensor: Indices of the k nearest neighbors for each tensor in the batch.
        """
        # Ensure tensors is on the same device as the queue
        tensors = tensors.to(self.device)

        # Compute cosine similarity for each tensor in the batch with the entire queue
        similarities = torch.nn.functional.cosine_similarity(
            tensors.unsqueeze(1),  # Shape: (batch_size, 1, dim)
            self.queue.unsqueeze(0),  # Shape: (1, size, dim)
            dim=2  # Compute similarity along the feature dimension
        )

        _, top_k_indices = torch.topk(similarities, k, dim=1)
        return top_k_indices
    
    def get_feats_at_indices(self, indices):
        """
        Get the features at the specified indices.

        Args:
            indices (torch.Tensor): Indices of the features to retrieve.

        Returns:
            torch.Tensor: Features at the specified indices.
        """
        return self.queue[indices]




class PRIMUSLightningModule(BasePretrainModule):
    def __init__(self, args):
        super().__init__(args)

        sensor_transforms = [
                noise_transform_vectorized, #0
                scaling_transform_vectorized, #1
                negate_transform_vectorized, #2
                time_flip_transform_vectorized, #3 
                time_segment_permutation_transform_improved, #4 
                rotation_transform_vectorized, #5
            ]

        self.sensor_transform = generate_combined_transform_function(sensor_transforms, self.hparams.transform_indices)

        self.mmcl_loss = InfoNCE(symmetric_loss=True, learn_temperature=True)
        self.ssl_loss = InfoNCE(symmetric_loss=True, learn_temperature=True)

        self.sensor_model = MW2StackRNNPoolingMultihead(num_sensors=self.hparams.num_sensors, size_embeddings=self.hparams.embedding_dim)
        self.video_model = Clip4CLIPModel(freeze=False)

         # ✅ t-SNE용 캐시 경로 (args에 tsne_cache_dir 넣어주면 됨)
        self.tsne_cache_dir = "/mnt/hdd4tb/junho/HWU-USP_v2/tsne_cache/primus_detailed_prudent"

        # ✅ validation에서 임시로 모아둘 버퍼들
        self.train_z_video = []
        self.train_z_sensor = []
        self.train_labels = []
        self.train_preds = []

    def setup(self, stage: str):
        if self.hparams.nnclr and stage == 'fit':
            self.vid_queue = FeatureQueue(8192, 512, device=self.device)
            self.sensor_queue = FeatureQueue(8192, 512, device=self.device)

    def fetch_from_queue(self, batch, domain='video'):

        videos = batch["video"]
        sensor_key = 'ssl_view=0'
        if domain == 'video':
            top_k_indices = self.vid_queue.find_nearest_neighbors(batch['video'], 1)
        elif domain == 'sensor':
            top_k_indices = self.sensor_queue.find_nearest_neighbors(batch[sensor_key], 1)
        else:
            raise ValueError('Invalid domain')

        bsz = videos.shape[0]
        vid_feats = self.vid_queue.get_feats_at_indices(top_k_indices.view(-1)).view(bsz, 512)
        sensor_feats = self.sensor_queue.get_feats_at_indices(top_k_indices.view(-1)).view(bsz, 512)

        # Update the queue
        self.vid_queue.enqueue(videos.detach())
        self.sensor_queue.enqueue(batch[sensor_key].detach())

        return vid_feats, sensor_feats


    def forward(self, batch, train_time=False):
        # x_sensor: (batch_size x 6 x window_size)
        # x_narration: [ str ] with len == batch_size
        # y_*: B x size_embeddings
        """
        if len(batch["video"]) != len(batch["narration"]) or len(batch["video"]) != len(batch["sensor"]):
            print("Weird!")
            min_size = min(min(len(batch["video"]), len(batch["narration"])), len(batch["sensor"]))
            batch["sensor"] = batch["sensor"][:min_size]
            batch["video"] = batch["video"][:min_size]
            batch["audio"] = batch["audio"][:min_size]
        """

        out = {}

        if train_time:
            for i in range(self.hparams.num_views):
                videos, sensors, labels, sample_ids, _ = batch
                if i == 0:
                    x_sensor = sensors
                else:
                    x_sensor = self.sensor_transform(sensors.cpu().numpy())
                    x_sensor = torch.Tensor(x_sensor).cuda() # INEFFICIENT!!!!

                y_sensor = self.sensor_model(x_sensor)

                out[f"ssl_view={i}"] = y_sensor["ssl"]
                out[f"mmcl_view={i}"] = y_sensor["mmcl"]
                out[f"emb_view={i}"] = y_sensor["emb"]

        else:
            x_sensor = sensors
            y_sensor = self.sensor_model(x_sensor)

            if self.multihead:
                out["ssl"] = y_sensor["ssl"]
                out["mmcl"] = y_sensor["mmcl"]
                out["emb"] = y_sensor["emb"]        
            else:
                out = y_sensor        

        x_video = videos
        y_video = torch.zeros((len(x_sensor), 512), dtype=torch.float32).to(self.device)
        for i in range(len(x_sensor)):
            videos, sensors, labels, sample_ids, _ = batch
            idx, sample_id = sample_ids
            # print(sample_id[0])
            # folder_name = sample_id[i].split('_')[0]  # "S3-ADL1"
            # cache_path = os.path.join(self.hparams.baseline_video_cache_dir, folder_name, sample_id[i] + '.pt')

            if False and os.path.exists(cache_path):
                y_video[i] = torch.load(cache_path).to(self.device)
            else:
                x = x_video[i]
                feat, _ = self.video_model.get_video_embeddings(x.unsqueeze(dim=0))
                y_video[i] = feat.squeeze(0)
                # os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                # torch.save(y_video[i].cpu().detach(), cache_path)
        out["video"] = y_video
        self.train_z_video.append(y_video.cpu().detach())
        self.train_z_sensor.append(y_sensor['emb'].cpu().detach())
        self.train_labels.append(labels.cpu().detach())
        self.train_preds.append(labels.cpu().detach())

        return out

    def training_step(self, batch, batch_idx: int):
          # y: {modality[str]: y_*} where y_*: B x size_embeddings
        print("training step batch idx:", batch_idx)
        y = self(batch, train_time=True)
        # Use NCE loss
        # y_query_modality = y[self.source_modality]
        loss_output = 0.0

        # MMCL Loss
        if self.hparams.ssl_coeff < 1:
            y_key_modality = y["video"]
            s2t_loss = self.mmcl_loss(query=y['mmcl_view=0'], positive_key=y_key_modality)

            # Log the loss
            str_s2t = "i2v"
            self.log(f"train_{str_s2t}_loss", s2t_loss, logger=True, sync_dist=True)
            loss_output += (1-self.hparams.ssl_coeff)*s2t_loss

        # SSL Loss
        if self.hparams.ssl_coeff > 0:
            for i in range(self.hparams.num_views-1):
                ssl_loss = self.ssl_loss(query=y[f"ssl_view=0"], positive_key=y[f"ssl_view={(i+1)}"])

                self.log("train_ssl_loss", ssl_loss, logger=True, sync_dist=True)
                # loss_output += (self.hparams.ssl_coeff)*ssl_loss


        if self.hparams.nnclr:
            ## NN-CLR Loss
            if self.hparams.ssl_coeff == 1:
                domain = 'sensor'
            else:
                domain = 'video'

            vid_feats, sensor_feats = self.fetch_from_queue(y, domain=domain)

            if self.hparams.ssl_coeff < 1: 
                    y_key_modality = vid_feats
                                        
                    s2t_loss = self.mmcl_loss(query=y['mmcl_view=0'], positive_key=y_key_modality)

                    # Log the loss
                    str_s2t = "i2v"
                    self.log(f"train_nn_{str_s2t}_loss", s2t_loss, logger=True, sync_dist=True)
                    loss_output += (1-self.hparams.ssl_coeff)*s2t_loss

            # SSL Loss
            if self.hparams.ssl_coeff > 0:
                for i in range(self.hparams.num_views-1):
                    ssl_loss = self.ssl_loss(query=y[f"ssl_view=0"], positive_key=sensor_feats)

                    self.log("train_nn_ssl_loss", ssl_loss, logger=True, sync_dist=True)
                    loss_output += (self.hparams.ssl_coeff)*ssl_loss 

    
        self.log("train_loss", loss_output, logger=True, sync_dist=True)
        return loss_output

    def on_train_epoch_end(self):
        if self.tsne_cache_dir is None:
            return

        if len(self.train_z_video) == 0:
            return

        # 1) concat
        z_video = torch.cat(self.train_z_video, dim=0).numpy()
        z_sensor = torch.cat(self.train_z_sensor, dim=0).numpy()
        labels  = torch.cat(self.train_labels, dim=0).numpy()
        preds   = torch.cat(self.train_preds, dim=0).numpy()

        # 2) 클래스 수 (라벨이 -1 이상인 경우만 사용)
        if (labels >= 0).any():
            num_clusters = int(labels[labels >= 0].max() + 1)
        else:
            num_clusters = -1

        # 3) 저장
        self.save_tsne_cache(
            save_dir=self.tsne_cache_dir,
            epoch=self.current_epoch,
            z_video=z_video,
            z_sensor=z_sensor,
            labels=labels,
            preds=preds,
            num_clusters=num_clusters,
            prefix="",  # 필요하면 "primus_" 같은 prefix 줄 수도 있음
        )

        # 4) 버퍼 비우기
        self.train_z_video.clear()
        self.train_z_sensor.clear()
        self.train_labels.clear()
        self.train_preds.clear()

    def save_tsne_cache(
        self,
        save_dir,
        epoch,
        z_video=None,
        z_sensor=None,
        labels=None,
        preds=None,
        num_clusters=None,
        prefix="",
    ):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{prefix}epoch_{epoch:04d}.npz")

        np.savez(
            save_path,
            z_video=z_video if z_video is not None else np.array([]),
            z_sensor=z_sensor if z_sensor is not None else np.array([]),
            labels=labels if labels is not None else np.array([]),
            preds=preds if preds is not None else np.array([]),
            num_clusters=np.array(num_clusters if num_clusters is not None else -1),
        )
        print(f"[t-SNE cache] Saved to {save_path}")

    def predict_step(self, batch, batch_idx: int):
        return self(batch)