import io
import hashlib
import os
from typing import Optional

import numpy as np
import cv2
from PIL import Image
import streamlit as st
import torch
from scipy.io import loadmat
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safetensors_load_file
from torchvision import transforms

from model import CSRNet
from utils import (
    load_image_from_bytes,
    preprocess_image_pil,
    tensor_density_to_numpy,
    density_to_overlay,
    density_from_points,
    MEAN,
    STD,
    IMG_SIZE,
    find_head_peaks,
    overlay_points_on_image,
    send_email,
)


st.set_page_config(page_title="CSRNet Crowd Counting", layout="wide")


def get_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _hash_bytes(data: Optional[bytes]) -> str:
    if not data:
        return "no-checkpoint"
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()[:16]


def _align_and_filter_state_dict(model: torch.nn.Module, state: dict):
    """Filter a checkpoint state_dict to only include keys that exist in the
    current model and have matching tensor shapes. Also strips an optional
    'module.' prefix from DataParallel checkpoints.

    Returns: (filtered_state, num_matched, num_skipped)
    """
    model_state = model.state_dict()
    filtered = {}
    matched = 0
    skipped = 0

    for k, v in state.items():
        key = k[7:] if k.startswith('module.') else k
        if key in model_state and model_state[key].shape == v.shape:
            filtered[key] = v
            matched += 1
        else:
            skipped += 1

    return filtered, matched, skipped


@st.cache_resource(show_spinner=False)
def load_model_cached(checkpoint_bytes: Optional[bytes], use_gpu: bool, variant: str, checkpoint_path: Optional[str] = None):
    device = get_device(use_gpu)
    # If a checkpoint is provided, avoid downloading pretrained weights and rely on the checkpoint
    init_with_pretrained = False if checkpoint_bytes else True
    model = CSRNet(load_weights=init_with_pretrained, variant=variant).to(device)

    if checkpoint_bytes or checkpoint_path:
        try:
            # Load state dict from bytes, safetensors, or a file path
            state = None
            if checkpoint_path and checkpoint_path.lower().endswith('.safetensors'):
                state = safetensors_load_file(checkpoint_path)
            else:
                if checkpoint_bytes:
                    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location=device)
                elif checkpoint_path:
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                else:
                    checkpoint = {}
                state = checkpoint.get("model_state_dict", checkpoint)
            # Filter incompatible keys and handle DP prefixes
            filtered, matched, skipped = _align_and_filter_state_dict(model, state)
            missing_in_ckpt = len([k for k in model.state_dict().keys() if k not in filtered])
            model.load_state_dict(filtered, strict=False)
            msg = f"Checkpoint loaded. Matched: {matched} | Skipped (shape/name mismatch): {skipped} | Missing in ckpt: {missing_in_ckpt}"
            st.sidebar.success(msg)
            if matched < 30:
                st.sidebar.warning("Few layers matched the checkpoint. If predictions look off, try switching the 'Checkpoint variant' between vgg16_bn and vgg16.")
        except Exception as e:
            st.sidebar.warning(f"Failed to load checkpoint: {e}")
    else:
        st.sidebar.info("No checkpoint provided. Using ImageNet-initialized frontend.")
        st.sidebar.warning("Predictions may be inaccurate without a trained checkpoint.")

    model.eval()
    return model, device


def run_inference(model: CSRNet, device: torch.device, image: Image.Image):
    with torch.inference_mode():
        input_tensor = preprocess_image_pil(image, device)
        density = model(input_tensor)
        count = float(density.sum().item())
        dm = tensor_density_to_numpy(density)
    return count, dm


def _preprocess_no_resize(img: Image.Image, device: torch.device) -> torch.Tensor:
    """Preprocess without resizing (ToTensor + Normalize)."""
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return t(img).unsqueeze(0).to(device)


def run_inference_tiled(model: CSRNet, device: torch.device, image: Image.Image, tile_size: int = 512, overlap: int = 64):
    """Run inference in overlapping tiles and stitch the density map back together.

    Each tile is processed at its native resolution (no resize), then the predicted
    density is resized to the tile size and blended into a canvas with averaging
    in overlapping regions.
    """
    w, h = image.size
    step = tile_size - overlap
    if step <= 0:
        step = tile_size

    density_canvas = np.zeros((h, w), dtype=np.float32)
    weight_canvas = np.zeros((h, w), dtype=np.float32)

    with torch.inference_mode():
        for y in range(0, h, step):
            for x in range(0, w, step):
                x2 = min(x + tile_size, w)
                y2 = min(y + tile_size, h)
                patch = image.crop((x, y, x2, y2))

                # If the patch is not exactly tile_size, pad to maintain consistent processing
                pad_w = tile_size - (x2 - x)
                pad_h = tile_size - (y2 - y)
                if pad_w > 0 or pad_h > 0:
                    padded = Image.new("RGB", (tile_size, tile_size))
                    padded.paste(patch, (0, 0))
                    patch = padded

                inp = _preprocess_no_resize(patch, device)
                out = model(inp)  # (1,1,h',w')
                dm_patch = tensor_density_to_numpy(out)
                # Resize predicted density back to the unpadded spatial size of this tile region
                dm_resized = cv2.resize(dm_patch, (tile_size, tile_size), interpolation=cv2.INTER_CUBIC)
                dm_crop = dm_resized[: (y2 - y), : (x2 - x)]

                density_canvas[y:y2, x:x2] += dm_crop
                weight_canvas[y:y2, x:x2] += 1.0

    weight_canvas[weight_canvas == 0] = 1.0
    stitched = density_canvas / weight_canvas
    total = float(stitched.sum())
    return total, stitched


def run_inference_multiscale(model: CSRNet, device: torch.device, image: Image.Image, scales: list[float]):
    """Run inference at multiple scales and average density maps after conserving mass.

    For each scale s, the image is resized to (IMG_SIZE * s) before inference.
    The predicted density is resized back to the original image size and
    renormalized to keep the total count consistent, then averaged over scales.
    """
    w, h = image.size
    dm_accum = np.zeros((h, w), dtype=np.float32)
    n = 0
    with torch.inference_mode():
        for s in scales:
            try:
                target_size = (max(32, int(IMG_SIZE[0] * s)), max(32, int(IMG_SIZE[1] * s)))
                t = transforms.Compose([
                    transforms.Resize(target_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEAN, std=STD),
                ])
                inp = t(image).unsqueeze(0).to(device)
                out = model(inp)
                dm = tensor_density_to_numpy(out)
                pre_sum = dm.sum() + 1e-6
                dm_resized = cv2.resize(dm, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
                post_sum = dm_resized.sum() + 1e-6
                dm_resized *= (pre_sum / post_sum)
                dm_accum += dm_resized
                n += 1
            except Exception:
                continue
    if n == 0:
        # Fallback to single-scale default
        return run_inference(model, device, image)
    dm_avg = dm_accum / float(n)
    total = float(dm_avg.sum())
    return total, dm_avg


def sidebar_controls():
    st.sidebar.header("GT-based Alert Settings")
    # Fixed GT directory (hidden from UI)
    gt_dir = r"e:/Downloads/archive/ShanghaiTech/part_A/test_data/ground-truth"
    st.sidebar.caption("Using configured GT directory.")
    threshold = st.sidebar.slider("Alert threshold (people)", min_value=1, max_value=2000, value=50, step=1)

    st.sidebar.markdown("---")
    st.sidebar.caption("Email (SMTP)")
    enable_email = st.sidebar.checkbox("Enable email alerts", value=False)
    to_email = st.sidebar.text_input("To", value="", placeholder="recipient@example.com", disabled=not enable_email)
    from_email = st.sidebar.text_input("From email", value="", placeholder="your_email@gmail.com", disabled=not enable_email)
    smtp_server = st.sidebar.text_input("SMTP server", value="smtp.gmail.com", disabled=not enable_email)
    smtp_port = st.sidebar.number_input("SMTP port", min_value=1, max_value=65535, value=587, step=1, disabled=not enable_email)
    smtp_user = st.sidebar.text_input("SMTP username", value="", disabled=not enable_email)
    smtp_password = st.sidebar.text_input("SMTP password/app password", value="", type="password", disabled=not enable_email)

    email_cfg = {
        "gt_dir": gt_dir,
        "threshold": int(threshold),
        "enable_email": enable_email,
        "to_email": to_email,
        "from_email": from_email,
        "smtp_server": smtp_server,
        "smtp_port": int(smtp_port),
        "smtp_user": smtp_user,
        "smtp_password": smtp_password,
    }
    return email_cfg


def main():
    st.title("Crowd Count Alerts")
    st.caption("Import an image; the app shows the GT count and sends an alert if it crosses the threshold.")

    cfg = sidebar_controls()

    # Minimal styling
    st.markdown("<style>.card{padding:0.8rem 1rem;border:1px solid #e6e6e6;border-radius:8px;background:#fff;} .muted{color:#6b6b6b;font-size:0.9rem}</style>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Input Image")
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"]) 
        st.markdown("<span class='muted'>Or provide a local path</span>", unsafe_allow_html=True)
        local_path = st.text_input("Local image path", value="")
        run = st.button("Run", type="primary")

    image = None
    uploaded_name = None
    if local_path:
        if os.path.exists(local_path):
            try:
                image = Image.open(local_path).convert("RGB")
            except Exception as e:
                st.error(f"Failed to load image from path: {e}")
        else:
            st.warning("Provided local path does not exist. Please check the path.")
    if image is None and uploaded is not None:
        image = load_image_from_bytes(uploaded.read())
        try:
            uploaded_name = uploaded.name
        except Exception:
            uploaded_name = None

    if image is None:
        st.info("Upload an image or enter a valid local image path to begin.")
        return

    if not cfg["gt_dir"] or not os.path.isdir(cfg["gt_dir"]):
        st.error("Please set a valid Ground-truth directory in the sidebar.")
        return

    with col_left:
        st.image(image, caption="Input Image", use_column_width=True)

    # Locate GT file by image basename under fixed GT dir
    gt_count = None
    gt_path = None
    try:
        base = None
        if local_path:
            base = os.path.splitext(os.path.basename(local_path))[0]
        elif uploaded_name:
            base = os.path.splitext(os.path.basename(uploaded_name))[0]
        if base:
            mat_path = os.path.join(cfg["gt_dir"], f"GT_{base}.mat")
            if os.path.exists(mat_path):
                gt_path = mat_path
                data = loadmat(gt_path)
                pts = None
                try:
                    pts = data['image_info'][0, 0][0, 0][0]
                except Exception:
                    for v in data.values():
                        if hasattr(v, 'shape') and len(v.shape) == 2 and v.shape[1] == 2:
                            pts = v
                            break
                if pts is not None:
                    gt_count = int(pts.shape[0])
                    # Quick overlay (points to density for visualization)
                    w, h = image.size
                    dm_gt = density_from_points((w, h), pts, sigma=15.0)
                    overlay = density_to_overlay(image, dm_gt)
                    with col_right:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.subheader("Result")
                        st.metric(label="Ground-truth Count", value=int(gt_count))
                        st.image(overlay, caption="Density Overlay (GT)", use_column_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not read ground-truth: {e}")

    if gt_count is None:
        st.error("Ground-truth not found for this image. Check the GT directory and file naming.")
        return

    # Auto email if threshold crossed
    if cfg.get("enable_email") and cfg.get("to_email"):
        try:
            if gt_count >= cfg["threshold"]:
                st.info(f"Threshold crossed (count={gt_count} ≥ {cfg['threshold']}). Sending email…")
                attach_path = os.path.join("alerts_out", "current_overlay.jpg")
                os.makedirs(os.path.dirname(attach_path), exist_ok=True)
                overlay.save(attach_path)
                send_email(
                    subject=f"Crowd Alert: {gt_count} people detected",
                    body=f"GT crowd count is {gt_count} for the current image.",
                    to_email=cfg["to_email"],
                    from_email=cfg["from_email"],
                    smtp_server=cfg["smtp_server"],
                    smtp_port=cfg["smtp_port"],
                    smtp_user=cfg["smtp_user"],
                    smtp_password=cfg["smtp_password"],
                    attachment_path=attach_path,
                )
                st.success(f"Alert email sent to {cfg['to_email']}")
        except Exception as e:
            st.warning(f"Email not sent: {e}")

    # Status card
    with col_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Status")
        st.write(f"- Threshold: {cfg['threshold']}")
        if cfg.get("enable_email") and cfg.get("to_email"):
            masked = cfg['to_email'][:2] + "***" + cfg['to_email'].split('@')[-1]
            st.write(f"- Email alerts: On → {masked}")
        else:
            st.write("- Email alerts: Off")
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
