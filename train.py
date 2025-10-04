import os
import argparse
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import CSRNet



class ShanghaiTechDataset(Dataset):
    def __init__(self, root: str, split: str = "train", img_size: Tuple[int, int] = (512, 512)):
        """
        Args:
            root: path to part_A/ or part_B/ directory that contains train_data/ and test_data/
            split: 'train' or 'test'
            img_size: resize images for training/inference
        Expected structure (example):
            <root>/train_data/images/IMG_1.jpg
            <root>/train_data/ground-truth/GT_IMG_1.mat
        """
        assert split in {"train", "test"}
        self.root = root
        self.split = split
        self.img_size = img_size

        split_dir = os.path.join(root, f"{split}_data")
        self.images_dir = os.path.join(split_dir, "images")
        # GT folder can be 'ground-truth' or 'ground_truth'
        self.gt_dir = os.path.join(split_dir, "ground-truth")
        if not os.path.exists(self.gt_dir):
            self.gt_dir = os.path.join(split_dir, "ground_truth")

        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"Images folder not found: {self.images_dir}")
        if not os.path.isdir(self.gt_dir):
            raise FileNotFoundError(f"Ground-truth folder not found: {self.gt_dir}")

        self.img_files = sorted([f for f in os.listdir(self.images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        if len(self.img_files) == 0:
            raise RuntimeError(f"No images found in {self.images_dir}")

        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_files)

    def _read_points_from_mat(self, mat_path: str) -> np.ndarray:
        data = loadmat(mat_path)
        points = None
        # Typical ShanghaiTech index path
        try:
            points = data['image_info'][0, 0][0, 0][0]
        except Exception:
            for v in data.values():
                if hasattr(v, 'shape') and len(v.shape) == 2 and v.shape[1] == 2:
                    points = v
                    break
        if points is None:
            return np.zeros((0, 2))
        return points

    def _points_to_density(self, size_wh: Tuple[int, int], points: np.ndarray, sigma: float = 15.0) -> np.ndarray:
        w, h = size_wh
        dm = np.zeros((h, w), dtype=np.float32)
        for p in points:
            x, y = int(p[0]), int(p[1])
            if 0 <= x < w and 0 <= y < h:
                dm[y, x] = 1.0
        dm = gaussian_filter(dm, sigma=sigma)
        return dm

    def __getitem__(self, idx: int):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mat_name = f"GT_{os.path.splitext(img_name)[0]}.mat"
        gt_path = os.path.join(self.gt_dir, mat_name)

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # Read GT points and build density map in original size, then resize to model output size later
        points = self._read_points_from_mat(gt_path)
        density = self._points_to_density((w, h), points, sigma=15.0)

        # Apply image transform
        image_t = self.transform(image)

        # Resize density to match image_t spatial size (H, W)
        tgt_h, tgt_w = image_t.shape[1], image_t.shape[2]
        density_resized = cv2.resize(density, (tgt_w, tgt_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        # Keep the total count approximately consistent after resize
        original_sum = density.sum() + 1e-6
        resized_sum = density_resized.sum() + 1e-6
        density_resized *= (original_sum / resized_sum)

        target_t = torch.from_numpy(density_resized).float()  # (H, W)
        return image_t, target_t.unsqueeze(0)  # (1, H, W)



class CrowdCountingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: (N, 1, H, W), target: (N, 1, H, W)
        if pred.shape != target.shape:
            target = torch.nn.functional.interpolate(target, size=pred.shape[2:], mode='bilinear', align_corners=False)
        return self.mse(pred, target)


def mae_rmse(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    pred_c = pred.sum(dim=(1, 2, 3))
    tgt_c = target.sum(dim=(1, 2, 3))
    mae = torch.mean(torch.abs(pred_c - tgt_c)).item()
    rmse = torch.sqrt(torch.mean((pred_c - tgt_c) ** 2)).item()
    return mae, rmse


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    # Datasets
    train_ds = ShanghaiTechDataset(root=args.data_root, split='train', img_size=(args.img_size, args.img_size))
    val_ds = ShanghaiTechDataset(root=args.data_root, split='test', img_size=(args.img_size, args.img_size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Model: if resuming, avoid pretrained download, otherwise allow
    init_with_pretrained = False if args.resume and os.path.isfile(args.resume) else True
    model = CSRNet(load_weights=init_with_pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    criterion = CrowdCountingLoss()

    start_epoch = 1
    best_val = float('inf')

    # Resume from checkpoint
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state, strict=False)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        model.train()
        running_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        train_loss = running_loss / max(1, len(train_loader))

        # Validate
        model.eval()
        val_loss = 0.0
        val_mae_total = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                mae, _ = mae_rmse(outputs, targets)
                val_mae_total += mae
        val_loss /= max(1, len(val_loader))
        val_mae = val_mae_total / max(1, len(val_loader))

        scheduler.step(val_loss)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_MAE={val_mae:.2f} lr={optimizer.param_groups[0]['lr']:.6f}")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            save_path = os.path.join(args.out_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            print(f"Saved best checkpoint: {save_path}")

        # Save periodic
        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(args.out_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(args.out_dir, 'final_model.pth')
    torch.save({'epoch': args.epochs, 'model_state_dict': model.state_dict()}, final_path)
    print(f"Training finished. Final model: {final_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train CSRNet on ShanghaiTech")
    p.add_argument('--data-root', dest='data_root', type=str, required=True,
                   help='Path to part_A or part_B directory that contains train_data/ and test_data/')
    p.add_argument('--out-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', dest='batch_size', type=int, default=4)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--img-size', type=int, default=512)
    p.add_argument('--save-interval', type=int, default=10)
    p.add_argument('--resume', type=str, default='', help='Path to a checkpoint to resume from')
    p.add_argument('--use-gpu', action='store_true', help='Use CUDA if available')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
