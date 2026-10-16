#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Defines the model architecture and training logic for the Slice Predictor.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
import pytorch_lightning as pl
import timm
from torchmetrics import SpearmanCorrCoef, AUROC, RetrievalHitRate
from torchvision import transforms

def log(m): print(f"[Model Log] {m}", flush=True)

class SlicePredictor(pl.LightningModule):
    """
    PyTorch Lightning module for the Vision Transformer-based slice predictor.
    """

    def __init__(self,
                 model_name: str = "vit_base_patch16_224.augreg_in21k",
                 num_slices: int = 240,
                 learning_rate: float = 1e-4,
                 backbone_learning_rate: float = 1e-5,
                 backbone_frozen: bool = False,
                 data_image_size: int = 256,
                 cache_dir: str = None):
        super().__init__()
        self.save_hyperparameters()

        # 1) Backbone
        log(f"Attempting to create model: {model_name}")
        log(f"Using cache directory: {cache_dir}")
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=3,
            num_classes=0,
            cache_dir=self.hparams.cache_dir
        )
        log("Successfully created backbone.")

        # 2) Freeze or finetune
        if self.hparams.backbone_frozen:
            log("Backbone is FROZEN. Training in feature-extraction mode.")
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        else:
            log("Backbone is FINETUNED. Training in end-to-end mode.")

        # 3) Resize to model input size if needed
        model_cfg = self.backbone.default_cfg
        self.model_input_size = (model_cfg['input_size'][1], model_cfg['input_size'][2])
        data_size = (data_image_size, data_image_size)
        if self.model_input_size != data_size:
            log(f"Creating resize transform for {data_size} -> {self.model_input_size}")
            self.resize_transform = transforms.Resize(self.model_input_size, antialias=True)
        else:
            self.resize_transform = nn.Identity()

        # 4) Head
        feature_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.hparams.num_slices),
            nn.Sigmoid()  # scores in [0,1]
        )

        # 5) Loss
        self.loss_fn = nn.MSELoss()

        # 6) Metrics
        # AUROC (micro over slices)
        self.val_auc = AUROC(task="binary")
        self.test_auc = AUROC(task="binary")

        # Retrieval HitRate@K
        self.val_hit_rate_1 = RetrievalHitRate(top_k=1)
        self.val_hit_rate_3 = RetrievalHitRate(top_k=3)
        self.val_hit_rate_5 = RetrievalHitRate(top_k=5)
        self.val_hit_rate_10 = RetrievalHitRate(top_k=10)
        self.val_hit_rate_20 = RetrievalHitRate(top_k=20)

        self.test_hit_rate_1 = RetrievalHitRate(top_k=1)
        self.test_hit_rate_3 = RetrievalHitRate(top_k=3)
        self.test_hit_rate_5 = RetrievalHitRate(top_k=5)
        self.test_hit_rate_10 = RetrievalHitRate(top_k=10)
        self.test_hit_rate_20 = RetrievalHitRate(top_k=20)

        # Per-study Spearman accumulators (non-persistent so old ckpts load fine)
        self.register_buffer("val_spearman_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("val_spearman_count", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("test_spearman_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("test_spearman_count", torch.tensor(0, dtype=torch.long), persistent=False)
        # Macro AR@10; non-persistent to preserve checkpoint compatibility.
        self.register_buffer("val_ar10_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("val_ar10_count", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("test_ar10_sum", torch.tensor(0.0), persistent=False)
        self.register_buffer("test_ar10_count", torch.tensor(0, dtype=torch.long), persistent=False)
        self._spearman_impl = SpearmanCorrCoef()

        # Global group-id counters for retrieval metrics (reset each epoch)
        self._val_gid = 0
        self._test_gid = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resize_transform(x)
        if self.hparams.backbone_frozen:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        return self.head(features)

    # ----------------------- Training -----------------------

    def training_step(self, batch, batch_idx):
        image_tensor, gt_vector, _ = batch
        pred_vector = self(image_tensor)
        loss = self.loss_fn(pred_vector, gt_vector)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    # ----------------------- Helpers ------------------------

    def _update_per_study_spearman(self, pred_vector: torch.Tensor, gt_vector: torch.Tensor, split: str):
        """
        Compute Spearman for each study (length-num_slices vector) and accumulate sum/count.
        Skip NaNs (e.g., constant targets).
        """
        B = gt_vector.shape[0]
        self._spearman_impl = self._spearman_impl.to(pred_vector.device)
        vals = []
        for i in range(B):
            s = self._spearman_impl(pred_vector[i], gt_vector[i])
            if not torch.isnan(s):
                vals.append(s)
        if vals:
            add = torch.stack(vals).sum()
            cnt = torch.tensor(len(vals), device=pred_vector.device, dtype=torch.long)
            if split == "val":
                self.val_spearman_sum += add
                self.val_spearman_count += cnt
            else:
                self.test_spearman_sum += add
                self.test_spearman_count += cnt

    def _update_ar_at_k(self, pred_vector: torch.Tensor, gt_vector: torch.Tensor, k: int, split: str):
        """
        Average Recall@K (AR@K) per CT:
        recall_i = |topK(pred_i) ∩ Pos_i| / |Pos_i|
        Macro-average across CTs with |Pos_i|>0.
        """
        B, S = gt_vector.shape
        k = min(k, S)

        # Top-K indices per study
        topk_idx = torch.topk(pred_vector, k=k, dim=1, largest=True, sorted=False).indices  # [B,k]

        # Positive mask and counts
        pos_mask = (gt_vector > 0)                # [B,S]
        pos_count = pos_mask.sum(dim=1)           # [B]
        valid = pos_count > 0

        if not valid.any():
            return

        # Hits in top-K
        hits = pos_mask.gather(dim=1, index=topk_idx).sum(dim=1)  # [B]
        recall = hits[valid].float() / pos_count[valid].float()   # [n_valid]

        if split == "val":
            self.val_ar10_sum += recall.sum()
            self.val_ar10_count += valid.sum()
        else:
            self.test_ar10_sum += recall.sum()
            self.test_ar10_count += valid.sum()


    # ----------------------- Validation ---------------------

    def on_validation_epoch_start(self):
        # reset per-epoch accumulators / counters
        self.val_spearman_sum.zero_()
        self.val_spearman_count.zero_()
        self.val_ar10_sum.zero_()
        self.val_ar10_count.zero_()
        self._val_gid = 0

    def validation_step(self, batch, batch_idx):
        image_tensor, gt_vector, _ = batch
        B = image_tensor.shape[0]
        S = self.hparams.num_slices

        pred_vector = self(image_tensor)

        # Loss
        loss = self.loss_fn(pred_vector, gt_vector)
        self.log('val_loss', loss, prog_bar=True, logger=True)

        # AUC (micro)
        preds_flat = pred_vector.flatten()
        target_flat = gt_vector.flatten()
        target_bool = (target_flat > 0).int()
        self.val_auc.update(preds_flat, target_bool)

        # Retrieval HitRate@K with globally unique group IDs across the epoch
        base_ids = torch.arange(self._val_gid, self._val_gid + B, device=self.device)
        indexes = torch.repeat_interleave(base_ids, S)
        self._val_gid += B

        self.val_hit_rate_1.update(preds_flat, target_bool, indexes=indexes)
        self.val_hit_rate_3.update(preds_flat, target_bool, indexes=indexes)
        self.val_hit_rate_5.update(preds_flat, target_bool, indexes=indexes)
        self.val_hit_rate_10.update(preds_flat, target_bool, indexes=indexes)
        self.val_hit_rate_20.update(preds_flat, target_bool, indexes=indexes)

        # Per-study Spearman accumulation
        self._update_per_study_spearman(pred_vector, gt_vector, split="val")
        self._update_ar_at_k(pred_vector, gt_vector, k=10, split="val")

        # epoch-level logs (metrics themselves handle compute on_epoch=True)
        self.log('val_auc_epoch', self.val_auc, on_step=False, on_epoch=True)
        self.log('val_hit_rate_1_epoch', self.val_hit_rate_1, on_step=False, on_epoch=True)
        self.log('val_hit_rate_3_epoch', self.val_hit_rate_3, on_step=False, on_epoch=True)
        self.log('val_hit_rate_5_epoch', self.val_hit_rate_5, on_step=False, on_epoch=True)
        self.log('val_hit_rate_10_epoch', self.val_hit_rate_10, on_step=False, on_epoch=True)
        self.log('val_hit_rate_20_epoch', self.val_hit_rate_20, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        # mean per-study Spearman
        if self.val_spearman_count > 0:
            mean_s = (self.val_spearman_sum / self.val_spearman_count).detach()
        else:
            mean_s = torch.tensor(float('nan'), device=self.device)
        self.log('val_spearman_epoch', mean_s, prog_bar=True)
        if self.val_ar10_count > 0:
            mean_ar10 = (self.val_ar10_sum / self.val_ar10_count).detach()
        else:
            mean_ar10 = torch.tensor(float('nan'), device=self.device)
        self.log('val_ar_10_epoch', mean_ar10)

    # ----------------------- Test ---------------------------

    def on_test_epoch_start(self):
        # reset per-epoch accumulators / counters
        self.test_spearman_sum.zero_()
        self.test_spearman_count.zero_()
        self.test_ar10_sum.zero_()
        self.test_ar10_count.zero_()
        self._test_gid = 0

    def test_step(self, batch, batch_idx):
        image_tensor, gt_vector, _ = batch
        B = image_tensor.shape[0]
        S = self.hparams.num_slices

        pred_vector = self(image_tensor)

        # Loss
        loss = self.loss_fn(pred_vector, gt_vector)
        self.log('test_loss', loss, logger=True)

        # AUC (micro)
        preds_flat = pred_vector.flatten()
        target_flat = gt_vector.flatten()
        target_bool = (target_flat > 0).int()
        self.test_auc.update(preds_flat, target_bool)

        # Retrieval HitRate@K with globally unique group IDs across the epoch
        base_ids = torch.arange(self._test_gid, self._test_gid + B, device=self.device)
        indexes = torch.repeat_interleave(base_ids, S)
        self._test_gid += B

        self.test_hit_rate_1.update(preds_flat, target_bool, indexes=indexes)
        self.test_hit_rate_3.update(preds_flat, target_bool, indexes=indexes)
        self.test_hit_rate_5.update(preds_flat, target_bool, indexes=indexes)
        self.test_hit_rate_10.update(preds_flat, target_bool, indexes=indexes)
        self.test_hit_rate_20.update(preds_flat, target_bool, indexes=indexes)

        # Per-study Spearman accumulation
        self._update_per_study_spearman(pred_vector, gt_vector, split="test")
        self._update_ar_at_k(pred_vector, gt_vector, k=10, split="test")

        # epoch-level logs
        self.log('test_auc_epoch', self.test_auc, on_step=False, on_epoch=True)
        self.log('test_hit_rate_1_epoch', self.test_hit_rate_1, on_step=False, on_epoch=True)
        self.log('test_hit_rate_3_epoch', self.test_hit_rate_3, on_step=False, on_epoch=True)
        self.log('test_hit_rate_5_epoch', self.test_hit_rate_5, on_step=False, on_epoch=True)
        self.log('test_hit_rate_10_epoch', self.test_hit_rate_10, on_step=False, on_epoch=True)
        self.log('test_hit_rate_20_epoch', self.test_hit_rate_20, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        if self.test_spearman_count > 0:
            mean_s = (self.test_spearman_sum / self.test_spearman_count).detach()
        else:
            mean_s = torch.tensor(float('nan'), device=self.device)
        self.log('test_spearman_epoch', mean_s, prog_bar=True)
        if self.test_ar10_count > 0:
            mean_ar10 = (self.test_ar10_sum / self.test_ar10_count).detach()
        else:
            mean_ar10 = torch.tensor(float('nan'), device=self.device)
        self.log('test_ar_10_epoch', mean_ar10, prog_bar=True)

    # ----------------------- Optimizer ----------------------

    def configure_optimizers(self):
        if self.hparams.backbone_frozen:
            log("Configuring optimizer for HEAD ONLY.")
            optimizer = AdamW(self.head.parameters(), lr=self.hparams.learning_rate)
        else:
            log(f"Configuring optimizer with differential LRs: Head={self.hparams.learning_rate}, Backbone={self.hparams.backbone_learning_rate}")
            param_groups = [
                {"params": self.head.parameters(), "lr": self.hparams.learning_rate},
                {"params": self.backbone.parameters(), "lr": self.hparams.backbone_learning_rate},
            ]
            optimizer = AdamW(param_groups)
        return optimizer


# --- Example of how to create the model ---
if __name__ == "__main__":
    def log_test(m): print(f"[Test Log] {m}")

    log_test("--- Creating Model 1: ImageNet Baseline (ViT-Base) ---")
    model_baseline = SlicePredictor(
        model_name="vit_base_patch16_224.augreg_in21k",
        num_slices=240,
        backbone_frozen=False,
        learning_rate=1e-4,
        backbone_learning_rate=1e-5,
        data_image_size=256
    )
    log_test("...Success.")

    log_test("--- Creating Model 2: ImageNet-22k (Swin-Tiny) ---")
    model_swin = SlicePredictor(
        model_name="swin_tiny_patch4_window7_224.ms_in22k",
        num_slices=240,
        backbone_frozen=False,
        learning_rate=1e-4,
        backbone_learning_rate=1e-5,
        data_image_size=256
    )
    log_test("...Success.")

    log_test("--- Creating Model 3: Medical Foundation Model (DINO) ---")
    model_radfm = SlicePredictor(
        model_name="hf_hub:Snarcy/RadioDino-s8",
        num_slices=240,
        backbone_frozen=False,
        learning_rate=1e-4,
        backbone_learning_rate=1e-5,
        data_image_size=256
    )
    log_test("...Success.")

    dummy_image = torch.randn(2, 3, 256, 256)
    log_test(f"\nTesting forward pass with {model_baseline.hparams.model_name}...")
    output = model_baseline(dummy_image)
    log_test(f"Output shape: {output.shape}")

    log_test(f"Testing forward pass with {model_radfm.hparams.model_name}...")
    output_radfm = model_radfm(dummy_image)
    log_test(f"Output shape: {output_radfm.shape}")

    log_test("\nModel architecture is ready.")
