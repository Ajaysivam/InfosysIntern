import io
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from scipy.ndimage import maximum_filter
from PIL import ImageDraw
import smtplib
from email.message import EmailMessage
from scipy.io import loadmat


IMG_SIZE = (512, 512)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def load_image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def preprocess_image_pil(image: Image.Image, device: torch.device) -> torch.Tensor:
    transform = get_val_transform()
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor


def density_to_overlay(original: Image.Image, density_map: np.ndarray, alpha: float = 0.35) -> Image.Image:
    """Create a heatmap overlay from a density map on top of the original image.

    Args:
        original: PIL RGB image
        density_map: HxW numpy array
        alpha: blending factor for heatmap
    Returns:
        PIL.Image with overlay
    """
    # Normalize density map to [0, 1]
    dm = density_map.astype(np.float32)
    if dm.max() > dm.min():
        dm = (dm - dm.min()) / (dm.max() - dm.min())
    else:
        dm = np.zeros_like(dm)

    # Resize density to original size
    w, h = original.size
    dm_resized = cv2.resize(dm, (w, h), interpolation=cv2.INTER_CUBIC)

    # Apply colormap (matplotlib-like 'hot')
    heatmap = cv2.applyColorMap((dm_resized * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend
    base = np.array(original).astype(np.float32)
    overlay = (1 - alpha) * base + alpha * heatmap.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


def tensor_density_to_numpy(density_tensor: torch.Tensor) -> np.ndarray:
    """Convert a model output (1x1xHxW) tensor to HxW numpy array on CPU."""
    dm = density_tensor.detach().squeeze().cpu().numpy()
    return dm


def density_from_points(image_size: tuple[int, int], points: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Create a density map (H, W) from point annotations using a Gaussian filter.

    Args:
        image_size: (width, height) of the target density map
        points: numpy array of shape (N, 2) with (x, y) coordinates
        sigma: Gaussian sigma in pixels
    Returns:
        Density map as float32 numpy array of shape (H, W)
    """
    w, h = image_size
    dm = np.zeros((h, w), dtype=np.float32)
    if points is None or len(points) == 0:
        return dm
    for p in points:
        x, y = int(p[0]), int(p[1])
        if 0 <= x < w and 0 <= y < h:
            dm[y, x] = 1.0
    dm = gaussian_filter(dm, sigma=sigma)
    return dm


def find_head_peaks(
    density_map: np.ndarray,
    min_distance: int = 4,
    threshold_rel: float = 0.1,
    smoothing_sigma: float = 1.0,
) -> np.ndarray:
    """Detect head centers as local maxima in a density map using non-maximum suppression.

    Args:
        density_map: float32 array (H, W)
        min_distance: minimum neighborhood distance between peaks (in pixels)
        threshold_rel: relative threshold in [0,1] applied to max(density_map)
        smoothing_sigma: optional Gaussian smoothing to reduce spurious peaks
    Returns:
        points: (N, 2) array of (x, y) integer pixel coordinates
    """
    if density_map is None or density_map.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    dm = density_map.astype(np.float32)
    if smoothing_sigma and smoothing_sigma > 0:
        dm = gaussian_filter(dm, sigma=float(smoothing_sigma))

    # Compute absolute threshold
    maxv = float(dm.max()) if dm.size else 0.0
    if maxv <= 0:
        return np.zeros((0, 2), dtype=np.int32)
    thr = float(threshold_rel) * maxv

    # Non-maximum suppression via maximum_filter over a square window
    size = int(max(1, 2 * min_distance + 1))
    local_max = maximum_filter(dm, size=size, mode='nearest')
    peaks_mask = (dm == local_max) & (dm >= thr)

    ys, xs = np.nonzero(peaks_mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    points = np.stack([xs.astype(np.int32), ys.astype(np.int32)], axis=1)
    return points


def overlay_points_on_image(
    original: Image.Image,
    points: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    radius: int = 3,
) -> Image.Image:
    """Draw small circles at given (x, y) points on the original image.

    Args:
        original: PIL RGB image
        points: (N, 2) array of (x, y)
        color: RGB tuple for points
        radius: circle radius in pixels
    Returns:
        PIL.Image with point annotations
    """
    img = original.copy()
    draw = ImageDraw.Draw(img)
    if points is not None:
        for (x, y) in points:
            x0, y0 = int(x - radius), int(y - radius)
            x1, y1 = int(x + radius), int(y + radius)
            draw.ellipse([x0, y0, x1, y1], outline=color, width=2)
    return img


def send_email(
    subject: str,
    body: str,
    to_email: str,
    from_email: str | None = None,
    smtp_server: str | None = None,
    smtp_port: int | None = None,
    smtp_user: str | None = None,
    smtp_password: str | None = None,
    attachment_path: str | None = None,
):
    """Send an email using SMTP with optional attachment.

    Falls back to environment variables if parameters are not provided:
      FROM_EMAIL, SMTP_SERVER (default smtp.gmail.com), SMTP_PORT (default 587),
      SMTP_USER, SMTP_PASSWORD
    """
    import os

    from_email = from_email or os.environ.get("FROM_EMAIL")
    smtp_server = smtp_server or os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(smtp_port or os.environ.get("SMTP_PORT", "587"))
    smtp_user = smtp_user or os.environ.get("SMTP_USER")
    smtp_password = smtp_password or os.environ.get("SMTP_PASSWORD")

    if not (from_email and smtp_user and smtp_password):
        raise RuntimeError("Missing SMTP credentials: set FROM_EMAIL, SMTP_USER, SMTP_PASSWORD (env vars) or pass them in.")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content(body)

    if attachment_path and os.path.isfile(attachment_path):
        with open(attachment_path, "rb") as f:
            data = f.read()
        msg.add_attachment(data, maintype="application", subtype="octet-stream", filename=os.path.basename(attachment_path))

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        return True


def read_shanghaitech_points_from_mat(mat_path: str) -> np.ndarray:
    """Read GT points from a ShanghaiTech-style .mat file robustly.

    Supports common variants:
      - data['image_info'][0,0]['location'][0,0] -> (N,2)
      - data['image_info'][0,0][0,0][0] -> (N,2)
      - any array value of shape (N,2)
    Returns (N,2) float32 array or empty array if not found.
    """
    try:
        data = loadmat(mat_path)
    except Exception:
        return np.zeros((0, 2), dtype=np.float32)

    pts = None
    try:
        if 'image_info' in data:
            info = data['image_info']
            # Case 1: has 'location' field
            try:
                loc = info[0, 0]['location']
                # sometimes another layer [0,0]
                if isinstance(loc, np.ndarray) and loc.size > 0 and isinstance(loc[0, 0], np.ndarray):
                    pts = loc[0, 0]
                else:
                    pts = loc
            except Exception:
                # Case 2: older indexing
                try:
                    pts = info[0, 0][0, 0][0]
                except Exception:
                    pts = None
    except Exception:
        pts = None

    # Fallback: scan for a (N,2)-shaped array among values
    if pts is None:
        for v in data.values():
            if isinstance(v, np.ndarray) and len(v.shape) == 2 and v.shape[1] == 2:
                pts = v
                break

    if pts is None:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    return pts
