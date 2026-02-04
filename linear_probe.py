import os
import argparse
import torch
import random
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from pathlib import Path # pathlib 임포트

# --- 사용자 정의 모듈 임포트 ---
from datamodule import MethodDataModule
from method import MethodLightningModule
from method_utils import gather

####################################################################
#                         Utility Functions
####################################################################

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def profile_model_efficiency(model, datamodule, args, device='cuda'):
    """
    GFLOPS와 FPS(Throughput) 측정
    """
    import time
    
    model.eval()
    model.to(device)
    
    # 샘플 데이터 준비
    datamodule.setup("fit")
    sample_batch = next(iter(datamodule.val_dataloader()))
    
    # 데이터 형태에 따라 처리
    if len(sample_batch) == 5:
        video_data, sensor_data, labels, _, flow = sample_batch
    else:
        video_data, sensor_data, labels = sample_batch[:3]
        flow = None
    
    sensor_data = sensor_data.to(device)
    video_data = video_data.to(device) if video_data is not None else None
    
    # flow가 dict인 경우 처리
    if flow is not None:
        if isinstance(flow, dict):
            flow = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in flow.items()}
        elif isinstance(flow, torch.Tensor):
            flow = flow.to(device)
        # 그 외의 경우 그대로 둠
    
    batch_size = sensor_data.shape[0]
    
    print("\n" + "=" * 70)
    print("Model Efficiency Profiling")
    print("=" * 70)
    
    # ============================================================
    # 1. FLOPs 계산 (fvcore 또는 thop 사용)
    # ============================================================
    flops_giga = None
    encoder_fn = None  # throughput 측정용
    
    # LinearProbeLightningModule의 경우
    if hasattr(model, 'model'):
        backbone = model.model
        
        # sensor encoder 선택 및 forward 함수 설정
        if args.model_name == "method":
            sensor_encoder = backbone.sensor_model
            encoder_fn = lambda x: sensor_encoder(x)
        elif args.model_name == "mae":
            # MAE는 forward_sensor_only 메서드만 사용
            mae_model = backbone.model
            sensor_encoder = mae_model  # 파라미터 카운트용
            encoder_fn = lambda x: mae_model.forward_sensor_only(x)
            
            # FLOPs 계산을 위한 래퍼 모듈
            class SensorOnlyWrapper(nn.Module):
                def __init__(self, mae):
                    super().__init__()
                    self.mae = mae
                def forward(self, x):
                    return self.mae.forward_sensor_only(x)
            sensor_encoder_for_flops = SensorOnlyWrapper(mae_model).to(device)
        else:
            sensor_encoder = getattr(backbone, 'sensor_model', backbone)
            encoder_fn = lambda x: sensor_encoder(x)
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in sensor_encoder.parameters())
        print(f"[Encoder] Total Params: {total_params:,} ({total_params/1e6:.2f}M)")
        
        # FLOPs 계산 시도 (fvcore 우선 + 상세 breakdown)
        try:
            from fvcore.nn import FlopCountAnalysis, flop_count_table
            
            if args.model_name == "mae":
                target_encoder = sensor_encoder_for_flops
            else:
                target_encoder = sensor_encoder
            
            target_encoder.eval()
            with torch.no_grad():
                flops_analyzer = FlopCountAnalysis(target_encoder, (sensor_data,))
                flops_analyzer.unsupported_ops_warnings(False)
                flops = flops_analyzer.total()
                
                # 상세 breakdown 출력
                print("\n--- FLOPs Breakdown by Module ---")
                by_module = flops_analyzer.by_module()
                # Top-level 모듈만 출력
                for name, val in by_module.items():
                    if val > 0 and name.count('.') <= 1:  # 1단계 깊이만
                        print(f"  {name}: {val/1e6:.3f} MFLOPs")
                
            flops_giga = flops / 1e9
            print(f"\n[Encoder] Total FLOPs (fvcore): {flops_giga:.4f} GFLOPs")
            
        except ImportError:
            print("[INFO] fvcore not installed. Install with: pip install fvcore")
            flops_giga = None
        except Exception as e:
            print(f"[WARNING] FLOPs calculation failed: {e}")
            flops_giga = None
        
        # 수동 FLOPs 계산 (보조)
        print("\n--- Manual FLOPs Estimation ---")
        manual_flops = 0
        for name, module in sensor_encoder.named_modules():
            if isinstance(module, nn.Conv1d):
                # FLOPs = 2 * Cin * Cout * K * L_out
                cin, cout = module.in_channels, module.out_channels
                k = module.kernel_size[0]
                l_out = sensor_data.shape[-1] // (module.stride[0] if hasattr(module, 'stride') else 1)
                flops_conv = 2 * cin * cout * k * l_out * sensor_data.shape[0]
                manual_flops += flops_conv
                print(f"  Conv1d {name}: {flops_conv/1e6:.3f} MFLOPs")
            elif isinstance(module, nn.Conv2d):
                cin, cout = module.in_channels, module.out_channels
                k = module.kernel_size[0] * module.kernel_size[1]
                h_out = 14  # 대략적 추정
                w_out = 8
                flops_conv = 2 * cin * cout * k * h_out * w_out * sensor_data.shape[0]
                manual_flops += flops_conv
                print(f"  Conv2d {name}: {flops_conv/1e6:.3f} MFLOPs")
            elif isinstance(module, nn.Linear):
                cin, cout = module.in_features, module.out_features
                flops_linear = 2 * cin * cout * sensor_data.shape[0]
                manual_flops += flops_linear
                print(f"  Linear {name}: {flops_linear/1e6:.3f} MFLOPs")
            elif isinstance(module, nn.GRU):
                hidden = module.hidden_size
                inp = module.input_size
                seq_len = 32  # 대략적 추정
                # GRU: 3 gates, each with 2 matmuls
                flops_gru = 6 * (inp * hidden + hidden * hidden) * seq_len * sensor_data.shape[0]
                if module.bidirectional:
                    flops_gru *= 2
                manual_flops += flops_gru
                print(f"  GRU {name}: {flops_gru/1e6:.3f} MFLOPs")
            elif isinstance(module, nn.MultiheadAttention):
                embed_dim = module.embed_dim
                seq_len = 32
                # Attention: Q, K, V projections + attention scores + output projection
                flops_attn = 4 * embed_dim * embed_dim * seq_len * sensor_data.shape[0]
                flops_attn += 2 * seq_len * seq_len * embed_dim * sensor_data.shape[0]
                manual_flops += flops_attn
                print(f"  Attention {name}: {flops_attn/1e6:.3f} MFLOPs")
        
        print(f"\n[Encoder] Manual Total: {manual_flops/1e9:.4f} GFLOPs")
    
    # ============================================================
    # 2. Throughput (FPS) 측정 - Encoder만 측정
    # ============================================================
    print("\n--- Throughput Measurement (Encoder Only) ---")
    
    if encoder_fn is None:
        print("[WARNING] encoder_fn not defined, skipping throughput measurement")
        fps = None
        latency_ms = None
    else:
        # Warm-up (충분히 50회)
        print("Warming up GPU (50 iterations)...")
        with torch.no_grad():
            for _ in range(50):
                _ = encoder_fn(sensor_data)
        
        torch.cuda.synchronize()
        
        # 실제 측정 (500회로 늘림)
        num_iterations = 500
        
        # CUDA Events로 더 정확한 시간 측정
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = encoder_fn(sensor_data)
        end_event.record()
        
        # Wait for completion
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000  # ms to seconds
        
        total_samples = batch_size * num_iterations
        fps = total_samples / elapsed_time
        latency_ms = (elapsed_time / num_iterations) * 1000
        
        print(f"Batch Size: {batch_size}")
        print(f"Total Iterations: {num_iterations}")
        print(f"Total Time: {elapsed_time:.3f}s")
        print(f"Throughput (FPS): {fps:.2f} samples/sec")
        print(f"Latency per batch: {latency_ms:.3f} ms")
    
    # ============================================================
    # 3. 결과 요약
    # ============================================================
    print("\n" + "-" * 70)
    print("Efficiency Summary")
    print("-" * 70)
    if flops_giga is not None:
        print(f"GFLOPs:      {flops_giga:.3f}")
    print(f"FPS:         {fps:.2f}")
    print(f"Latency:     {latency_ms:.2f} ms")
    print("=" * 70 + "\n")
    
    return {
        'gflops': flops_giga,
        'fps': fps,
        'latency_ms': latency_ms
    }


def set_module_params(args):
    try:
        # 경로를 Path 객체로 변환
        path = Path(args.checkpoint_path)
                
        parent_dir = path.parent

        args.ckpt_name = path.parent / path.stem

        
        if 'manual_epochs' in parent_dir.name:
            # 3-A. 부모가 'manual_epochs'인 경우 (예: .../model/dataset/manual_epochs/ckpt)
            dataset_dir = parent_dir.parent
            model_dir = dataset_dir.parent
            
            args.dataset_name = dataset_dir.name
            args.model_name = model_dir.name
        else:
            # 3-B. 부모가 'manual_epochs'가 아닌 경우 (예: .../model/dataset/ckpt)
            dataset_dir = parent_dir
            model_dir = dataset_dir.parent
            
            args.dataset_name = dataset_dir.name
            args.model_name = model_dir.name

        print(f"Dataset: {args.dataset_name}, Model: {args.model_name}, Ckpt: {args.ckpt_name}")
    
    except Exception as e:
        print(f"경로 분석 중 오류 발생: {e}")
        print("경로 구조가 예상과 다릅니다. (예: .../model_name/dataset_name/[manual_epochs]/ckpt_name.ckpt)")
        
    return args

def get_backbone_with_mode(args):
    # 기존 pretrained checkpoint 로드 루틴
    backbone = load_pretrained_model(args)
        # ⚠️ 반드시 device 이동 이후에 freeze
    if hasattr(backbone, "to"):
        print("move to cuda")
        backbone = backbone.to("cuda")
    
    # backbone.train()
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # 모델 내 첫 번째 BN 레이어를 찾는 예시
    for name, m in backbone.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            print(f"Layer: {name}")
            print(f" - Running Mean (첫 5개): {m.running_mean[:5]}")
            print(f" - Running Var (첫 5개): {m.running_var[:5]}")
            break

    # ✅ 3️⃣ Gradient 상태 확인 (디버깅용)
    total_params = sum(1 for _ in backbone.parameters())
    trainable_params = sum(p.requires_grad for p in backbone.parameters())
    print(f"[Backbone Grad Check] Trainable parameters: {trainable_params}/{total_params}")
    if trainable_params:
        sample_layers = [n for n, p in backbone.named_parameters() if p.requires_grad][:5]
        print(f"→ Sample trainable layers: {sample_layers}")
    else:
        print("✅ All parameters are frozen (no grad flow to backbone).")

    return backbone


def load_pretrained_model(args):
    ckpt = args.checkpoint_path
    if not ckpt or not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"--- Loading pretrained model from: {ckpt} ---")

    if args.model_name == "method":
        from method import MethodLightningModule
        model = MethodLightningModule.load_from_checkpoint(ckpt, strict=False)
    elif args.model_name == "comodo":
        from baseline_modules.comodo.module import COMODOLightningModule
        model = COMODOLightningModule.load_from_checkpoint(ckpt, strict=False, map_location='cpu').to('cuda')
    elif args.model_name == "primus":
        from baseline_modules.primus import PRIMUSLightningModule
        model = PRIMUSLightningModule.load_from_checkpoint(ckpt)
    elif args.model_name == "imu2clip":
        from baseline_modules.imu2clip import IMU2CLIPLightningModule
        model = IMU2CLIPLightningModule.load_from_checkpoint(ckpt)
    elif args.model_name == "mae":
        from baseline_modules.mae import EVIMAELightningModule
        model = EVIMAELightningModule.load_from_checkpoint(ckpt, map_location='cpu').to('cuda')
    else:
        raise ValueError(f"Unknown model_name: {args.model_name}")
    
    # --- Debug: Print Loaded Parameter Structure ---
    print(f"\n[Debug] Verifying loaded parameters for model: {args.model_name}")
    param_keys = list(model.state_dict().keys())
    print(f"[Debug] Total keys in model state_dict: {len(param_keys)}")
    prefixes = set()
    for k in param_keys:
        parts = k.split('.')
        if len(parts) >= 2:
            prefixes.add(f"{parts[0]}.{parts[1]}")
        else:
            prefixes.add(parts[0])
    print("[Debug] Model parameter groups (prefixes):")
    for p in sorted(list(prefixes)):
        print(f"  - {p}")
    print("-" * 50 + "\n")
    # -----------------------------------------------

    return model


####################################################################
#                        Linear Probe Module
####################################################################

class LinearProbeLightningModule(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = model

        # for param in self.model.parameters():
        #     param.requires_grad = False

        emb_dim = model.hparams.embedding_dim
        if self.hparams.model_name == "method":
            emb_dim *= 2
        elif self.hparams.model_name == "comodo":
            emb_dim //= 2

        # encoder_type에 따라 embedding dimension 조정
        self.encoder_type = getattr(self.hparams, 'encoder_type', 'sensor')
        if self.encoder_type == "sensor-video":
            emb_dim *= 2  # sensor + video 임베딩 concatenation

        self.classifier = torch.nn.Linear(emb_dim, self.hparams.num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
 # 5. Loss 함수 및 정확도 메트릭 정의
 
        # F1 Scores (Micro, Macro, Weighted)
        self.val_f1_micro = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='micro')
        self.val_f1_macro = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='macro')
        self.val_f1_weighted = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='weighted')
        
        self.test_f1_micro = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='micro')
        self.test_f1_macro = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='macro')
        self.test_f1_weighted = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='weighted')

        # mAUC (Multiclass AUROC) - average='macro'가 일반적
        self.val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=self.hparams.num_classes, average='macro')
        self.test_auroc = torchmetrics.AUROC(task="multiclass", num_classes=self.hparams.num_classes, average='macro')

        # mAP (Multiclass Average Precision) - average='macro'가 일반적
        self.val_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=self.hparams.num_classes, average='macro')
        self.test_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=self.hparams.num_classes, average='macro')

        # --- ✅ Confusion Matrix 추가 ---
        self.val_conf_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_conf_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.hparams.num_classes)


        # Confusion matrix용 클래스 이름
        self.class_dic = {
            0: 'Open Door 1', 1: 'Open Door 2', 2: 'Close Door 1', 3: 'Close Door 2',
            4: 'Open Fridge', 5: 'Close Fridge', 6: 'Open Dishwasher', 7: 'Close Dishwasher',
            8: 'Open Drawer 1', 9: 'Close Drawer 1', 10: 'Open Drawer 2',
            11: 'Close Drawer 2', 12: 'Open Drawer 3', 13: 'Close Drawer 3'
        }
        self.class_names = [v for v in self.class_dic.values()]

    def _get_sensor_embedding(self, sensor_data):
        """센서 인코더에서 임베딩 추출"""
        if self.hparams.model_name == "method":
            sensor_encoder = self.model.sensor_model
            return sensor_encoder(sensor_data)
        elif self.hparams.model_name == "imu2clip":
            sensor_encoder = self.model.sensor_model
            sensor_data = self.model.sensor_padding(sensor_data)
            return sensor_encoder(sensor_data)
        elif self.hparams.model_name == "primus":
            sensor_encoder = self.model.sensor_model
            return sensor_encoder(sensor_data)['mmcl']
        elif self.hparams.model_name == "mae":
            return self.model.model.forward_sensor_only(sensor_data)
        elif self.hparams.model_name == "comodo":
            sensor_encoder = self.model.sensor_model
            return sensor_encoder(sensor_data)
        else:
            raise ValueError(f"Unknown model_name for sensor: {self.hparams.model_name}")

    def _get_video_embedding(self, video_data, flow=None):
        """비디오 인코더에서 임베딩 추출 (inference_mode로 메모리 절약)"""
        if self.hparams.model_name == "method":
            video_encoder = self.model.video_model
            # method는 flow를 두번째 인자로 받음
            return video_encoder(video_data, flow)["z_video_online"]
        elif self.hparams.model_name == "imu2clip":
            video_encoder = self.model.video_model
            # imu2clip은 (B, T, C, H, W) -> (B, C, T, H, W) 변환 필요
            video_data = video_data.permute(0, 2, 1, 3, 4)
            return video_encoder.get_video_embeddings(video_data)
        elif self.hparams.model_name == "primus":
            video_encoder = self.model.video_model
            video_emb, _ = video_encoder.get_video_embeddings(video_data)
            return video_emb
        elif self.hparams.model_name == "mae":
            return self.model.model.forward_video_only(video_data)
        elif self.hparams.model_name == "comodo":
            # COMODO는 video_teacher를 사용
            video_encoder = self.model.video_teacher
            return video_encoder.encode(video_data)
        else:
            raise ValueError(f"Unknown model_name for video: {self.hparams.model_name}")

    def forward(self, video_data, sensor_data, flow=None):
        # encoder_type에 따라 다른 임베딩 사용
        if self.encoder_type == "sensor":
            representations = self._get_sensor_embedding(sensor_data)
        elif self.encoder_type == "video":
            representations = self._get_video_embedding(video_data, flow)
        elif self.encoder_type == "sensor-video":
            sensor_emb = self._get_sensor_embedding(sensor_data)
            video_emb = self._get_video_embedding(video_data, flow)
            representations = torch.cat([sensor_emb, video_emb], dim=1)
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        logits = self.classifier(representations)
        
        # ✅ gradient 흐름 확인 (1회만 출력)
        if self.current_epoch == 0:
            if self.hparams.model_name == "mae":
                encoder = self.model.model
            else:
                encoder = self.model.sensor_model
            grad_flags = [
                (n, p.requires_grad, p.grad is not None)
                for n, p in encoder.named_parameters()
            ]
            trainable = [n for n, rg, _ in grad_flags if rg]
            print(f"[Gradient Check] Trainable params in encoder: {len(trainable)}")
            total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
            print(f"Total trainable parameters in encoder: {total_params:,}")
            print(f"→ Sample trainable layers: {trainable[:5]}")
        
        return logits

    def _shared_step(self, batch, batch_idx):
        video_data, sensor_data, y, _, flow = batch
        logits = self(video_data, sensor_data, flow)
        loss = self.criterion(logits, y)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, probs, y

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        print("grad: ", loss.requires_grad)
        return loss

    # def on_after_backward(self):
    #     # 1️⃣ sensor encoder 파라미터 중 grad 있는 개수 카운트
    #     try:
    #         encoder = (
    #             self.model.model.sensor_model
    #             if hasattr(self.model, "model") and hasattr(self.model.model, "sensor_model")
    #             else self.model.sensor_model
    #         )

    #         grad_exist = []
    #         grad_sum = 0.0
    #         for name, p in encoder.named_parameters():
    #             if p.grad is not None:
    #                 grad_exist.append(name)
    #                 grad_sum += p.grad.abs().sum().item()

    #         print(f"[Gradient Flow] params_with_grad={len(grad_exist)}, grad_sum={grad_sum:.6f}")
    #         if grad_exist:
    #             print("→ sample layers:", grad_exist[:5])
    #     except Exception as e:
    #         print(f"Error in on_after_backward: {e}")


    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_labels = []

    def validation_step(self, batch, batch_idx):
        loss, preds, probs, y = self._shared_step(batch, batch_idx)
        self.val_accuracy.update(preds, y)
        self.val_f1_micro.update(preds, y)
        self.val_f1_macro.update(preds, y)
        self.val_f1_weighted.update(preds, y)
        self.val_auroc.update(probs, y) # AUC는 확률값(probs) 사용
        self.val_ap.update(probs, y)    # AP는 확률값(probs) 사용
        # ✅ confusion matrix용 raw 저장
        self.val_preds.append(preds.detach().cpu())
        self.val_labels.append(y.detach().cpu())

        self.log("val/val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/val_acc", self.val_accuracy, on_epoch=True, prog_bar=True)
        self.log("val/val_f1_micro", self.val_f1_micro, on_step=False, on_epoch=True, prog_bar=False) # prog_bar는 선택사항
        self.log("val/val_f1_macro", self.val_f1_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/val_f1_weighted", self.val_f1_weighted, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/val_mAUC", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/val_mAP", self.val_ap, on_step=False, on_epoch=True, prog_bar=True)

        
    def on_validation_epoch_end(self):
        val_labels = gather(torch.cat(self.val_labels, dim=0))
        val_preds = gather(torch.cat(self.val_preds, dim=0))
        y_true = val_labels.cpu().numpy()
        y_pred = val_preds.cpu().numpy()

        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)  # 정규화 버전 (선택)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,                     # raw count
            annot=True, fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar=True, square=True,
            ax=ax
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Validation Confusion Matrix")

        # W&B에 업로드
        if self.logger is not None:
            self.logger.experiment.log({"Validation Confusion Matrix": wandb.Image(fig)})
        plt.close(fig)

    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_labels = []

    def test_step(self, batch, batch_idx):
        loss, preds, probs, y = self._shared_step(batch, batch_idx)
        self.test_accuracy.update(preds, y)
        self.test_f1_micro.update(preds, y)
        self.test_f1_macro.update(preds, y)
        self.test_f1_weighted.update(preds, y)
        self.test_auroc.update(probs, y)
        self.test_ap.update(probs, y)

        # ✅ raw 저장
        self.test_preds.append(preds.detach().cpu())
        self.test_labels.append(y.detach().cpu())

        self.log("test/loss", loss, on_epoch=True)
        self.log("test/acc", self.test_accuracy, on_epoch=True)
        self.log("test/f1_macro", self.test_f1_macro, on_epoch=True)
        self.log("test/f1_micro", self.test_f1_micro, on_step=False, on_epoch=True, prog_bar=False) # prog_bar는 선택사항
        self.log("test/f1_weighted", self.test_f1_weighted, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/mAUC", self.test_auroc, on_epoch=True)
        self.log("test/mAP", self.test_ap, on_epoch=True)

    def on_test_epoch_end(self):
        test_labels = gather(torch.cat(self.test_labels, dim=0))
        test_preds = gather(torch.cat(self.test_preds, dim=0))
        y_true = test_labels.cpu().numpy()
        y_pred = test_preds.cpu().numpy()

          # ✅ DDP 상태에서 모든 GPU의 결과를 모음
        # gathered_true = [torch.zeros_like(y_true) for _ in range(self.trainer.world_size)]
        # gathered_pred = [torch.zeros_like(y_pred) for _ in range(self.trainer.world_size)]
        # torch.distributed.all_gather(gathered_true, y_true)
        # torch.distributed.all_gather(gathered_pred, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Test Confusion Matrix")
        if self.logger is not None:
            self.logger.experiment.log({"Test Confusion Matrix": wandb.Image(fig)})
        plt.close(fig)


    def configure_optimizers(self):
        groups = [{'params': self.classifier.parameters(), 'lr': self.hparams.lr}]

        # supervision이면 sensor encoder도 함께 학습
        if getattr(self.hparams, 'supervision', False):
            enc_params = []
            for n, p in self.model.named_parameters():
                # requires_grad=True인 것만 (위에서 필터링 완료)
                if p.requires_grad:
                    enc_params.append(p)
            # 백본은 약간 낮은 LR 권장
            lr_backbone = getattr(self.hparams, 'lr_backbone', self.hparams.lr * 0.1)
            groups.append({'params': enc_params, 'lr': lr_backbone})

            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                print(f"[OPT] classifier lr={self.hparams.lr}, sensor_encoder lr={lr_backbone}, "
                    f"params={sum(p.numel() for g in groups for p in g['params'])}")

        return torch.optim.Adam(groups)

class LinearProbeLSTM(pl.LightningModule):
    """
    윈도우 임베딩(사전학습 센서 인코더) → LSTM → 시퀀스(액티비티) 분류
    """
    def __init__(self, args, backbone):
        super().__init__()
        self.save_hyperparameters(args)
        self.backbone = backbone  # 사전학습 모듈(PLModule)
        self.probe_mode = getattr(args, "probe_mode", "lstm")  # 기본값 lstm

        # 임베딩 차원: method만 *2, 그 외 그대로
        emb_dim = int(self.backbone.hparams.embedding_dim)
        if self.hparams.model_name == "method":
            emb_dim *= 2
            # self.backbone.hprams.embedding *=2
        
        # encoder_type에 따라 embedding dimension 조정
        self.encoder_type = getattr(args, 'encoder_type', 'sensor')
        if self.encoder_type == "sensor-video":
            emb_dim *= 2  # sensor + video 임베딩 concatenation
        
        self.emb_dim = emb_dim

             # 🔹 probe_mode가 LSTM일 때만 LSTM 정의
        if self.probe_mode == "lstm":
            self.temporal_model = nn.LSTM(
                input_size=self.emb_dim,
                hidden_size=self.emb_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
                dropout=0.3
            )
        elif self.probe_mode == "meanpool":
            self.temporal_model = None  # 단순 mean pooling 사용
        else:
            raise ValueError(f"Unknown probe_mode: {self.probe_mode}")

        # 교체 부분만 발췌
        # self.rnn = nn.GRU(
        #     input_size=self.emb_dim,
        #     hidden_size=self.emb_dim // 2,  # 더 약하게
        #     num_layers=1,
        #     batch_first=True,
        #     bidirectional=False,
        #     dropout=0.2,
        # )

        self.classifier = nn.Linear(self.emb_dim, self.hparams.num_classes)

        # Metrics
        self.criterion = nn.CrossEntropyLoss()
# -----------------------------
        # ✅ Metrics (VAL)
        # -----------------------------
        nc = self.hparams.num_classes
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=nc)
        self.val_f1_micro = torchmetrics.F1Score(task="multiclass", num_classes=nc, average='micro')
        self.val_f1_macro = torchmetrics.F1Score(task="multiclass", num_classes=nc, average='macro')
        self.val_f1_weighted = torchmetrics.F1Score(task="multiclass", num_classes=nc, average='weighted')
        self.val_prec_macro = torchmetrics.Precision(task="multiclass", num_classes=nc, average='macro')
        self.val_recall_macro = torchmetrics.Recall(task="multiclass", num_classes=nc, average='macro')
        self.val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=nc, average='macro')
        self.val_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=nc, average='macro')
        self.val_conf_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=nc)

        # -----------------------------
        # ✅ Metrics (TEST)
        # -----------------------------
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=nc)
        self.test_f1_micro = torchmetrics.F1Score(task="multiclass", num_classes=nc, average='micro')
        self.test_f1_macro = torchmetrics.F1Score(task="multiclass", num_classes=nc, average='macro')
        self.test_f1_weighted = torchmetrics.F1Score(task="multiclass", num_classes=nc, average='weighted')
        self.test_prec_macro = torchmetrics.Precision(task="multiclass", num_classes=nc, average='macro')
        self.test_recall_macro = torchmetrics.Recall(task="multiclass", num_classes=nc, average='macro')
        self.test_auroc = torchmetrics.AUROC(task="multiclass", num_classes=nc, average='macro')
        self.test_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes=nc, average='macro')
        self.test_conf_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=nc)

        self.val_preds_raw = []
        self.val_labels_raw = []
        self.test_preds_raw = []
        self.test_labels_raw = []
        

    def _encode_sensor_windows(self, windows_b_sct):
        """센서 윈도우 인코딩"""
        B, S, C, T = windows_b_sct.shape
        flat = windows_b_sct.reshape(B * S, C, T)

        if self.hparams.model_name in ["method", "comodo", "primus", "imu2clip"]:
            sensor_encoder = self.backbone.sensor_model

        if self.hparams.model_name == "method":
            reps = sensor_encoder(flat)
        elif self.hparams.model_name == "imu2clip":
            flat_padded = self.backbone.sensor_padding(flat)
            reps = sensor_encoder(flat_padded)
        elif self.hparams.model_name == "primus":
            reps = sensor_encoder(flat)['mmcl']
        elif self.hparams.model_name == "mae":
            reps = self.backbone.model.forward_sensor_only(flat)
        elif self.hparams.model_name == "comodo":
            reps = sensor_encoder(flat)
        else:
            raise ValueError(f"Unknown model_name for sensor: {self.hparams.model_name}")

        reps = reps.reshape(B, S, -1)
        return reps

    def _encode_video_windows(self, video_windows, flow_windows=None):
        """비디오 윈도우 인코딩 (정확히 batch_size개씩 처리하여 메모리 절약)"""
        B, S, T, C, H, W = video_windows.shape
        print("video_windows shape:", video_windows.shape)
        
        # B × S를 먼저 flatten
        total_windows = B * S
        all_videos = video_windows.reshape(total_windows, T, C, H, W)
        
        if flow_windows is not None:
            all_flows = flow_windows.reshape(total_windows, *flow_windows.shape[2:])
        else:
            all_flows = None
        
        chunk_size = self.hparams.batch_size  # 정확히 batch_size개씩 처리
        all_reps = []
        
        for start in range(0, total_windows, chunk_size):
            end = min(start + chunk_size, total_windows)
            chunk = all_videos[start:end].contiguous()
            
            if all_flows is not None:
                flow_chunk = all_flows[start:end].contiguous()
            else:
                flow_chunk = None
            
            with torch.inference_mode():
                if self.hparams.model_name == "method":
                    video_encoder = self.backbone.video_model
                    chunk_reps = video_encoder(chunk, flow_chunk)["z_video_online"]
                elif self.hparams.model_name == "imu2clip":
                    video_encoder = self.backbone.video_model
                    chunk = chunk.permute(0, 2, 1, 3, 4)
                    chunk_reps = video_encoder.get_video_embeddings(chunk)
                elif self.hparams.model_name == "primus":
                    video_encoder = self.backbone.video_model
                    chunk_reps, _ = video_encoder.get_video_embeddings(chunk)
                elif self.hparams.model_name == "mae":
                    chunk_reps = self.backbone.model.forward_video_only(chunk)
                elif self.hparams.model_name == "comodo":
                    video_encoder = self.backbone.video_teacher
                    chunk_reps = video_encoder.encode(chunk)
                else:
                    raise ValueError(f"Unknown model_name for video: {self.hparams.model_name}")
            
            all_reps.append(chunk_reps)
        
        reps = torch.cat(all_reps, dim=0)  # (total_windows, D)
        reps = reps.reshape(B, S, -1)  # 다시 (B, S, D)로 복원
        return reps

    def _encode_windows(self, sensor_windows, video_windows=None, flow_windows=None):
        """encoder_type에 따라 윈도우 인코딩"""
        if self.encoder_type == "sensor":
            return self._encode_sensor_windows(sensor_windows)
        elif self.encoder_type == "video":
            return self._encode_video_windows(video_windows, flow_windows)
        elif self.encoder_type == "sensor-video":
            sensor_reps = self._encode_sensor_windows(sensor_windows)
            video_reps = self._encode_video_windows(video_windows, flow_windows)
            return torch.cat([sensor_reps, video_reps], dim=-1)
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")
        
    def forward(self, sensor_windows, lengths_b, video_windows=None, flow_windows=None):
        seq_emb = self._encode_windows(sensor_windows, video_windows, flow_windows)  # [B, S, D]

        # 🔹 probe_mode별 처리
        if self.probe_mode == "meanpool":
            pooled = seq_emb.mean(dim=1)
            logits = self.classifier(pooled)
            return logits
        else:  # lstm
            packed = nn.utils.rnn.pack_padded_sequence(
                seq_emb, lengths_b.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.temporal_model(packed)
            h_last = h_n[-1]
            logits = self.classifier(h_last)
            return logits

    def _shared_step(self, batch):
        # batch에서 video 데이터와 flow도 가져오기 (datamodule에서 제공하는 경우)
        if len(batch) == 6:
            sensor_windows, video_windows, flow_windows, lengths, targets, metas = batch
            video_windows = video_windows.to(self.device, non_blocking=True) if video_windows is not None else None
            flow_windows = flow_windows.to(self.device, non_blocking=True) if flow_windows is not None else None
        elif len(batch) == 5:
            sensor_windows, video_windows, lengths, targets, metas = batch
            video_windows = video_windows.to(self.device, non_blocking=True) if video_windows is not None else None
            flow_windows = None
        else:
            sensor_windows, lengths, targets, metas = batch
            video_windows = None
            flow_windows = None
        
        sensor_windows = sensor_windows.to(self.device, non_blocking=True)
        lengths = lengths.to(self.device)
        targets = targets.to(self.device)

        logits = self(sensor_windows, lengths, video_windows, flow_windows)
        loss = self.criterion(logits, targets)
        preds = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)  # AUROC/AP 용
        return loss, preds, probs, targets

    # -----------------------------
    # Train
    # -----------------------------
    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    # -----------------------------
    # Validation
    # -----------------------------
    def on_validation_epoch_start(self):
        self.val_preds_raw.clear()
        self.val_labels_raw.clear()

    def validation_step(self, batch, batch_idx):
        loss, preds, probs, targets = self._shared_step(batch)
        # print("target ", targets)
        # 클래스 인덱스 기반 메트릭
        self.val_acc.update(preds, targets)
        self.val_f1_micro.update(preds, targets)
        self.val_f1_macro.update(preds, targets)
        self.val_f1_weighted.update(preds, targets)
        self.val_prec_macro.update(preds, targets)
        self.val_recall_macro.update(preds, targets)
        self.val_conf_matrix.update(preds, targets)

        # 확률 기반 메트릭
        self.val_auroc.update(probs, targets)
        self.val_ap.update(probs, targets)

        # 로깅/원시 저장
        self.val_preds_raw.append(preds.detach().cpu())
        self.val_labels_raw.append(targets.detach().cpu())

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        # print("Sample preds:", preds[:10].tolist())
        # print("Sample labels:", targets[:10].tolist())

    def on_validation_epoch_end(self):

        # scalar compute
        log_dict = {
            "val/acc": self.val_acc.compute(),
            "val/f1_micro": self.val_f1_micro.compute(),
            "val/f1_macro": self.val_f1_macro.compute(),
            "val/f1_weighted": self.val_f1_weighted.compute(),
            "val/precision_macro": self.val_prec_macro.compute(),
            "val/recall_macro": self.val_recall_macro.compute(),
            "val/auroc_macro": torch.nan_to_num(self.val_auroc.compute()),
            "val/ap_macro": torch.nan_to_num(self.val_ap.compute()),
        }
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)

        # Confusion Matrix 그림
        cm_t = self.val_conf_matrix.compute().cpu().numpy()  # [C, C]
        fig, ax = plt.subplots(figsize=(8, 6))
        class_names = self.class_names or [str(i) for i in range(cm_t.shape[0])]
        sns.heatmap(cm_t, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title("Validation Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        if self.logger:
            self.logger.experiment.log({"val/confusion_matrix": wandb.Image(fig)})
        plt.close(fig)

        # reset
        for m in [
            self.val_acc, self.val_f1_micro, self.val_f1_macro, self.val_f1_weighted,
            self.val_prec_macro, self.val_recall_macro, self.val_auroc, self.val_ap,
            self.val_conf_matrix
        ]:
            m.reset()

    # -----------------------------
    # Test
    # -----------------------------
    def on_test_epoch_start(self):
        self.test_preds_raw.clear()
        self.test_labels_raw.clear()

    def test_step(self, batch, batch_idx):
        loss, preds, probs, targets = self._shared_step(batch)

        # 클래스 인덱스 기반
        self.test_acc.update(preds, targets)
        self.test_f1_micro.update(preds, targets)
        self.test_f1_macro.update(preds, targets)
        self.test_f1_weighted.update(preds, targets)
        self.test_prec_macro.update(preds, targets)
        self.test_recall_macro.update(preds, targets)
        self.test_conf_matrix.update(preds, targets)

        # 확률 기반
        self.test_auroc.update(probs, targets)
        self.test_ap.update(probs, targets)

        self.test_preds_raw.append(preds.detach().cpu())
        self.test_labels_raw.append(targets.detach().cpu())

        self.log("test_loss", loss, on_epoch=True, sync_dist=True)

    def on_test_epoch_end(self):
        log_dict = {
            "test/acc": self.test_acc.compute(),
            "test/f1_micro": self.test_f1_micro.compute(),
            "test/f1_macro": self.test_f1_macro.compute(),
            "test/f1_weighted": self.test_f1_weighted.compute(),
            "test/precision_macro": self.test_prec_macro.compute(),
            "test/recall_macro": self.test_recall_macro.compute(),
            "test/auroc_macro": torch.nan_to_num(self.test_auroc.compute()),
            "test/ap_macro": torch.nan_to_num(self.test_ap.compute()),
        }
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)

        cm_t = self.test_conf_matrix.compute().cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 6))
        class_names = self.class_names or [str(i) for i in range(cm_t.shape[0])]
        sns.heatmap(cm_t, annot=True, fmt="d", cmap="Greens",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title("Test Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        if self.logger:
            self.logger.experiment.log({"test/confusion_matrix": wandb.Image(fig)})
        plt.close(fig)

        for m in [
            self.test_acc, self.test_f1_micro, self.test_f1_macro, self.test_f1_weighted,
            self.test_prec_macro, self.test_recall_macro, self.test_auroc, self.test_ap,
            # self.test_conf_matrix
        ]:
            m.reset()

    def configure_optimizers(self):
        # probe_mode별 optimizer 설정
        if self.probe_mode == "meanpool":
            params = list(self.classifier.parameters())
        else:
            params = list(self.temporal_model.parameters()) + list(self.classifier.parameters())
        return torch.optim.Adam(params, lr=self.hparams.lr)


def main(args):
    set_random_seed(42)
    args = set_module_params(args)

    # HWU-USP 분기
    if args.dataset_name == "HWU-USP":
        args.num_classes = 5
        args.threshold_epoch = -1
        args.seq_len = 100
        args.num_sensors = 6
        datamodule = MethodDataModule(args, stage="linear_probe_lstm")
        datamodule.setup("fit")  # ✅ 추가
        backbone = get_backbone_with_mode(args)
        model = LinearProbeLSTM(args, backbone)

        # ✅ class_to_idx를 datamodule에서 그대로 가져와 class_names에 저장
         # class_names 자동 추출
        class_names = list(datamodule.train_dataset.class_to_idx.keys())
        print(f"✅ Loaded class names: {class_names}")
        model.class_names = class_names
        monitor = 'val/acc'

    else:
        args.num_classes = 14
        args.threshold_epoch = -1
        args.seq_len = 128
        args.num_sensors = 37
        datamodule = MethodDataModule(args, stage='linear_probe')
        
        backbone = get_backbone_with_mode(args)
        model = LinearProbeLightningModule(args, backbone)
        monitor = 'val_acc'
    
    if args.encoder_type == "sensor":
        args.threshold_epoch = 100 # need only sensor data
        
    # 분산 환경이 초기화되었는지 확인
    if dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 4  # 초기화되지 않았으면 (즉, 단일 프로세스 실행) 1로 설정
    if args.sensor_only:
        run_name = f"{args.sensor_model_name}_{args.dataset_name}_linear_probe_{args.batch_size* world_size}_linearEpoch={args.linear_epochs}"
    else:
        sensor_tag = "_supervision" if args.supervision else ""
        run_name = f"{args.model_name}_{args.dataset_name}_{args.ckpt_name}_{backbone.hparams.batch_size*4}_{backbone.hparams.epochs}_linear_probe_{args.batch_size* world_size}_linearEpoch={args.linear_epochs}{sensor_tag}_{args.encoder_type}"
    is_master_process = os.environ.get("LOCAL_RANK", "0") == "0"
    logger = WandbLogger(project="Method_Linear_Probe", name=run_name) if is_master_process else False
    # ckpt_cb = ModelCheckpoint(monitor=monitor, mode='max',
    #                           dirpath=f'checkpoints_linear/{args.model_name}_{args.dataset_name}',
    #                           filename='best-{epoch:02d}-{val_acc:.3f}', save_top_k=1)

    trainer = pl.Trainer(
        max_epochs=args.linear_epochs,
        accelerator='gpu',
        devices=-1,
        strategy='ddp_find_unused_parameters_true',
        logger=logger,
        sync_batchnorm=True,
        # callbacks=[ckpt_cb]
    )

    # datamodule.setup("fit")

    # backbone.train()
    # with torch.no_grad():
    #     device = 'cuda'
    #     for i, batch in enumerate(datamodule.train_dataloader()):
    #         video_data, sensor_data, labels, _, flow = batch 
    #         _ = backbone.video_model(video_data.to(device), None)
    #         _ = backbone.sensor_model(sensor_data.to(device))
    #         print("update default setting...", i)
    #         if i == 200:
    #             break
    # backbone.eval()

    print("--- Starting Linear Probing ---")
    trainer.fit(model, datamodule)
    datamodule.setup("fit")

    # ============================================================
    # Efficiency Profiling (GFLOPS & FPS)
    # ============================================================
    if getattr(args, 'profile', False) and is_master_process:
        print("\n--- Running Efficiency Profiling ---")
        profile_model_efficiency(model, datamodule, args)

    print("--- Testing ---")
    trainer.test(model=model, dataloaders=datamodule.test_dataloader())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--linear_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--probe_mode', type=str, default='meanpool', choices=['lstm', 'meanpool'], help="Choose probing strategy: 'lstm' for temporal model or 'meanpool' for simple temporal average pooling")
    parser.add_argument('--supervision', type=bool, default=False, help='If True, skip backbone weight loading and run with frozen hyperparameter configs only')
    parser.add_argument('--sensor_only', action='store_true', help='Train sensor-only model from scratch (no checkpoint)')
    parser.add_argument('--sensor_model_name', type=str, default='dlinear', choices=['dlinear', 'timesnet', 'moment', 'mantis', 'imu2clip', 'comodo'])
    parser.add_argument('--embedding_dim', type=str, default=256, help='for sensor only mode')
    parser.add_argument('--encoder_type', type=str, default='sensor', choices=['video', 'sensor', 'sensor-video'], 
                        help='추론 시 사용할 인코더 타입: sensor(기본값), video, sensor-video(두 임베딩 concat)')
    parser.add_argument('--use_flow', action='store_true', help='Use optical flow')
    parser.add_argument('--profile', action='store_true', help='Run GFLOPS and FPS profiling after training')

    args = parser.parse_args()
    main(args)