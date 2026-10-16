#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Defines the PyTorch Dataset and DataModule for loading
the projection dataset.

This script handles loading the correct AIP, finding the
correct heatmap, and blending them together on-the-fly.
"""

import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from scipy.ndimage import gaussian_filter
from torchvision import transforms
import argparse 
from torchvision.utils import save_image

# --- Category definitions (from create_dataset.py) ---
CATS_TO_PROCESS = [
    "Atelectasis", "Consolidation", "Groundglass opacity",
    "High lung volume / emphysema", "Lung nodule or mass", "Pleural abnormality",
]
CAT2SLUG = {c: re.sub(r'[^a-z0-9]+','_',c.lower()).strip('_') for c in CATS_TO_PROCESS}
HEATMAP_SLUGS = {
    "Atelectasis": "atelectasis",
    "Consolidation": "consolidation",
    "Groundglass opacity": "groundglass-opacity",
    "High lung volume / emphysema": "high-lung-volume-emphysema", # <-- FIXED SLUG
    "Lung nodule or mass": "lung-nodule-or-mass",
    "Pleural abnormality": "pleural-abnormality",
}

# --- ImageNet Normalization constants ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class SliceDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the slice prediction data.
    """
    def __init__(self, 
                 data_root: Path, 
                 manifest: pd.DataFrame, 
                 heatmap_dir: Path,
                 projection_type: str, 
                 blend_alpha: float = 0.5, 
                 augment: bool = False,
                 target_mode: str = "area"):
        """
        Args:
            data_root (Path): Path to the `projection_dataset_final` folder.
            manifest (pd.DataFrame): The manifest for this split (train, val, etc.).
            heatmap_dir (Path): Path to the `avg_np` folder.
            projection_type (str): Name of the projection to load (e.g., "aip_wide").
            blend_alpha (float): The alpha value for blending the heatmap (0.0 to 1.0).
            augment (bool): Whether to apply data augmentation.
        """
        self.data_root = data_root
        self.manifest = manifest
        self.heatmap_dir = heatmap_dir
        self.projection_filename = f"{projection_type}.png"
        self.blend_alpha = blend_alpha
        self.augment = augment
        self.target_mode = target_mode
        
        # We only need to load heatmaps if we're blending
        self.heatmaps = {}
        if self.blend_alpha > 0 and self.heatmap_dir:
            self._preload_heatmaps()

        # Define data augmentations
        self.base_transform = transforms.Compose([
            # --- FIXED: Add a non-random flip to correct the orientation ---
            transforms.RandomHorizontalFlip(p=1.0), # Always flip horizontally
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        # --- UPDATED: No augmentations for this task ---
        self.train_augment = transforms.Compose([
            # We already have the flip in the base_transform
            self.base_transform
        ])
    def _presence_to_gaussian(self, presence: torch.Tensor) -> torch.Tensor:
        """
        Convert a 0/1 presence vector into a Gaussian-smoothed vector
        over each contiguous run of ones.

        presence: 1D tensor of shape (num_slices,) with values 0 or 1.
        Returns:  1D tensor of same shape, values in [0, 1].
        """
        # Ensure float
        presence = presence.float()
        out = torch.zeros_like(presence)

        # Indices where presence == 1
        ones_idx = (presence > 0.5).nonzero(as_tuple=False).view(-1)
        if ones_idx.numel() == 0:
            return out  # no lesion slices

        # Find contiguous runs
        start = None
        prev = None
        for idx in ones_idx:
            idx = idx.item()
            if start is None:
                start = idx
                prev = idx
            elif idx == prev + 1:
                prev = idx
            else:
                # close previous run
                self._fill_gaussian_run(out, start, prev)
                # start new run
                start = idx
                prev = idx

        # close final run
        if start is not None:
            self._fill_gaussian_run(out, start, prev)

        return out

    def _fill_gaussian_run(self, out: torch.Tensor, start: int, end: int):
        """
        Fill out[start:end+1] with a normalized 1D Gaussian.
        """
        L = end - start + 1
        if L <= 0:
            return
        # Create 1D positions
        x = torch.arange(L, dtype=torch.float32)
        center = (L - 1) / 2.0
        # Sigma proportional to run length, but at least 1.0
        sigma = max(L / 3.0, 1.0)
        g = torch.exp(-0.5 * ((x - center) / sigma) ** 2)
        if g.max() > 0:
            g = g / g.max()
        out[start:end+1] = g


    def _preload_heatmaps(self):
        """Loads and processes all heatmaps into memory for speed."""
        print(f"Pre-loading heatmaps from {self.heatmap_dir}...")
        for cat_name, heatmap_slug in HEATMAP_SLUGS.items():
            npz_path = self.heatmap_dir / f"gaze_raw_{heatmap_slug}.npz"
            if npz_path.is_file():
                try:
                    with np.load(npz_path) as data:
                        key = 'gaze_counts' if 'gaze_counts' in data else 'heat'
                        gaze_counts = data[key].astype(np.float32)
                    gaze_blurred = gaussian_filter(gaze_counts, sigma=10)
                    gaze_min, gaze_max = gaze_blurred.min(), gaze_blurred.max()
                    if gaze_max > gaze_min:
                        gaze_norm = (gaze_blurred - gaze_min) / (gaze_max - gaze_min)
                    else:
                        gaze_norm = np.zeros_like(gaze_blurred)
                    self.heatmaps[cat_name] = gaze_norm
                except Exception as e:
                    raise RuntimeError(f"Failed to load heatmap: {npz_path}") from e
            else:
                raise FileNotFoundError(f"Heatmap file not found: {npz_path}")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        # 1. Get sample info from the manifest
        row = self.manifest.iloc[idx]
        data_path = self.data_root / row['data_path']
        category = row['category']

        # 2. Load the base projection image (as 3-channel RGB)
        proj_path = data_path / self.projection_filename
        if not proj_path.is_file():
            raise FileNotFoundError(f"Projection not found: {proj_path}")
            
        base_image = Image.open(proj_path).convert("RGB")
        base_image_float = np.array(base_image).astype(np.float32) / 255.0 # (H, W, 3)

        # 3. Load the Ground Truth vector
        gt_data_path = data_path / "gt_data.json"
        if not gt_data_path.is_file():
            raise FileNotFoundError(f"Target file not found: {gt_data_path}")
            
        with open(gt_data_path, 'r') as f:
            gt_data = json.load(f)
        
        gt_vector = torch.tensor(gt_data['gt_vector'], dtype=torch.float32)
        gt_order = torch.tensor(gt_data['gt_order'], dtype=torch.int64)
        
        if self.target_mode == "area":
            # Original behavior: use area-normalized gt_vector as-is
            pass
        elif self.target_mode == "slice_binary":
            # 0/1 presence per slice (simulated slice-level labels)
            gt_vector = (gt_vector > 0).float()
        elif self.target_mode == "slice_gaussian":
            # First get binary presence, then apply Gaussian along runs
            presence = (gt_vector > 0).float()
            gt_vector = self._presence_to_gaussian(presence)
        else:
            raise ValueError(f"Unknown target_mode: {self.target_mode}")

        # 4. Handle Heatmap Blending (if alpha > 0)
        if self.blend_alpha > 0 and category in self.heatmaps:
            # Load pre-processed heatmap
            gaze_norm = self.heatmaps[category]
            
            # Load BBox coordinates
            bbox_path = data_path / "bbox_256.json"
            if not bbox_path.is_file():
                print(f"ERROR: Cannot find BBox JSON: {bbox_path}")
                # Continue without heatmap
            else:
                with open(bbox_path, 'r') as f:
                    bbox_data = json.load(f)
                
                x0_f, z0_f, x1_f, z1_f = bbox_data["bbox_256_xyxy"]
                bbox_w = int(np.round(x1_f - x0_f))
                bbox_h = int(np.round(z1_f - z0_f))

                if bbox_w > 0 and bbox_h > 0:
                    # Resize heatmap to fit BBox
                    gaze_resized = cv2.resize(gaze_norm, (bbox_w, bbox_h), interpolation=cv2.INTER_LINEAR)
                    
                    # --- Create the full-size alpha mask ---
                    full_alpha_mask = np.zeros((256, 256), dtype=np.float32)
                    x0_i, z0_i = int(np.round(x0_f)), int(np.round(z0_f))
                    
                    h_slice = slice(z0_i, min(z0_i + bbox_h, 256))
                    w_slice = slice(x0_i, min(x0_i + bbox_w, 256))
                    
                    h_heat = min(bbox_h, 256 - z0_i)
                    w_heat = min(bbox_w, 256 - x0_i)

                    if h_heat > 0 and w_heat > 0:
                        full_alpha_mask[h_slice, w_slice] = gaze_resized[:h_heat, :w_heat] * self.blend_alpha
                    
                    # --- Blend *only* the Red Channel ---
                    base_image_float[..., 0] = (base_image_float[..., 0] * (1.0 - full_alpha_mask)) + (1.0 * full_alpha_mask)
        
        # 5. Apply augmentations
        # Convert back to PIL Image for torchvision transforms
        final_image_pil = Image.fromarray((base_image_float * 255).astype(np.uint8), mode="RGB")
        
        if self.augment:
            image_tensor = self.train_augment(final_image_pil)
        else:
            image_tensor = self.base_transform(final_image_pil)
            
        # --- UPDATED: Return all three tensors ---
        return image_tensor, gt_vector, gt_order


class SliceDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule to wrap the SliceDataset.
    """
    def __init__(self, 
                 data_root: str, 
                 heatmap_dir: str,
                 splits_json: str,
                 projection_type: str,
                 blend_alpha: float = 0.5,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 exclude_categories_train: list[str] | None = None,      # NEW
                 test_only_category: str | None = None,
                 target_mode: str = "area"):                # NEW
        super().__init__()
        self.save_hyperparameters()
        
        self.data_root = Path(data_root)
        self.heatmap_dir = Path(heatmap_dir)
        self.splits_json = Path(splits_json)
        self.manifest_path = self.data_root / "manifest.csv"

    def setup(self, stage: str = None):
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        
        # Load the master manifest
        manifest_df = pd.read_csv(self.manifest_path)
        
        # Load the split assignments
        with open(self.splits_json, 'r') as f:
            splits = json.load(f)
        
        train_cts = set(splits.get('train', []))
        val_cts = set(splits.get('val', []))
        test_cts = set(splits.get('test', []))

         # Filter manifest by split
        self.train_df = manifest_df[manifest_df['ct_name'].isin(train_cts)]
        self.val_df   = manifest_df[manifest_df['ct_name'].isin(val_cts)]
        self.test_df  = manifest_df[manifest_df['ct_name'].isin(test_cts)]

        # --- NEW: Leave-One-Category-Out controls ---
        # 1) Exclude selected categories from TRAIN only
        if self.hparams.exclude_categories_train:
            excl = set(self.hparams.exclude_categories_train)
            before = len(self.train_df)
            self.train_df = self.train_df[~self.train_df['category'].isin(excl)]
            print(f"Excluded from TRAIN: {sorted(excl)} "
                  f"(removed {before - len(self.train_df)} rows)")

        # 2) Evaluate zero-shot: restrict TEST to a single category (optional)
        if self.hparams.test_only_category:
            target = self.hparams.test_only_category
            before = len(self.test_df)
            self.test_df = self.test_df[self.test_df['category'] == target]
            print(f"TEST restricted to category: {target} "
                  f"(kept {len(self.test_df)} / {before} rows)")

        # Create datasets
        self.train_dataset = SliceDataset(
        self.data_root, self.train_df, self.heatmap_dir,
        self.hparams.projection_type, self.hparams.blend_alpha,
        augment=True,
        target_mode=self.hparams.target_mode
        )
        self.val_dataset = SliceDataset(
            self.data_root, self.val_df, self.heatmap_dir,
            self.hparams.projection_type, self.hparams.blend_alpha,
            augment=False,
            target_mode=self.hparams.target_mode
        )
        self.test_dataset = SliceDataset(
            self.data_root, self.test_df, self.heatmap_dir,
            self.hparams.projection_type, self.hparams.blend_alpha,
            augment=False,
            target_mode=self.hparams.target_mode
        )
        
        print(f"Dataset setup complete:")
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Val samples:   {len(self.val_dataset)}")
        print(f"  Test samples:  {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

# --- __main__ block for testing and VISUALIZING the dataloader ---
if __name__ == "__main__":
    
    # This block allows you to run `python dataset.py` to test it
    
    parser = argparse.ArgumentParser(description="Test SliceDataModule")
    
    parser.add_argument('--data_root', type=str, default='./projection_dataset_final',
                        help='Root directory of the final dataset')
    parser.add_argument('--heatmap_dir', type=str, default='./avg_np',
                        help='Directory with raw REFLACX heatmaps')
    parser.add_argument('--splits_json', type=str, default='new_balanced_splits.json',
                        help='Name of the splits file (e.g., new_balanced_splits.json)')
    parser.add_argument('--projection_type', type=str, default='aip_wide',
                        help='Which AIP projection to use as input.')
    parser.add_argument('--blend_alpha', type=float, default=0.5,
                        help='Alpha for heatmap blending (e.g., 0.0 for baseline, 0.5 for blend)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0,
                        help="Set to 0 for local testing on Windows")

    args = parser.parse_args()

    print("--- Initializing DataModule for Test ---")
    
    # Use the full path for the splits_json
    splits_path = Path(args.data_root) / args.splits_json
    if not splits_path.is_file():
        # Fallback for running on local PC where splits might be in the root
        splits_path = Path(args.splits_json)
        if not splits_path.is_file():
            print(f"ERROR: Cannot find {args.splits_json} in {args.data_root} or current dir.")
            sys.exit(1)


    datamodule = SliceDataModule(
        data_root=args.data_root,
        heatmap_dir=args.heatmap_dir,
        splits_json=str(splits_path),
        projection_type=args.projection_type,
        blend_alpha=args.blend_alpha,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print("\nRunning datamodule.setup()...")
    datamodule.setup()
    
    print("\n--- Testing Validation Dataloader (augment=False) ---")
    loader = datamodule.val_dataloader()
    
    # --- Function to de-normalize images for saving ---
    def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        # (B, C, H, W) -> (B, C, H, W)
        mean = torch.tensor(mean).view(1, 3, 1, 1)
        std = torch.tensor(std).view(1, 3, 1, 1)
        tensor = tensor * std + mean  # De-normalize
        tensor = torch.clamp(tensor, 0, 1) # Clamp to [0, 1]
        return tensor

    # Get just the first batch
    try:
        batch = next(iter(loader))
        # --- UPDATED: Unpack all three items ---
        image_tensor, gt_vector_tensor, gt_order_tensor = batch
        
        print("\nBatch 1:")
        # Image Tensor
        print(f"  Image Tensor Shape: {image_tensor.shape}")
        print(f"  Image Tensor Type: {image_tensor.dtype}")
        print(f"  Image Tensor Min (Normalized): {image_tensor.min():.2f}, Max (Normalized): {image_tensor.max():.2f}")
        
        # GT Vector
        print(f"  GT Vector Shape: {gt_vector_tensor.shape}")
        print(f"  GT Vector Type: {gt_vector_tensor.dtype}")
        print(f"  GT Vector Min: {gt_vector_tensor.min():.2f}, Max: {gt_vector_tensor.max():.2f}")

        # --- NEW: Check GT Order ---
        print(f"  GT Order Shape: {gt_order_tensor.shape}")
        print(f"  GT Order Type: {gt_order_tensor.dtype}")
        print(f"  GT Order Example (first 10 vals): {gt_order_tensor[0, :10]}")

        # --- Save the batch as an image grid ---
        # De-normalize the tensor
        image_tensor_denorm = denormalize(image_tensor)
        
        # Save to a grid
        save_path = "dataloader_test_batch.png"
        save_image(image_tensor_denorm, save_path, nrow=args.batch_size)
        print(f"\n--- Saved visualization of first validation batch to {save_path} ---")

    except Exception as e:
        print(f"\n--- ERROR during dataloader test ---")
        print(f"Could not load batch. This might be a path error.")
        print(f"Make sure paths are correct:")
        print(f"  --data_root: {args.data_root}")
        print(f"  --heatmap_dir: {args.heatmap_dir}")
        print(f"  --splits_json: {splits_path}")
        print(f"\nError details: {e}")

    print("\n--- Dataloader Test Complete ---")
