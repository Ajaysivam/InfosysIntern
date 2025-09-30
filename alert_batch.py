import os
import argparse
from typing import Optional, Tuple
import csv

import numpy as np
from PIL import Image

from utils import (
    density_from_points,
    overlay_points_on_image,
    send_email,
    read_shanghaitech_points_from_mat,
)


def _read_points_from_mat(mat_path: str) -> np.ndarray:
    return read_shanghaitech_points_from_mat(mat_path)


def process_image(
    img_path: str,
    gt_dir: str,
    out_dir: Optional[str] = None,
) -> Tuple[int, Optional[str]]:
    """Return crowd count from ground truth for an image, and optional path to overlay.
    Count is taken as number of GT points.
    """
    img_name = os.path.basename(img_path)
    name_wo = os.path.splitext(img_name)[0]
    mat_name = f"GT_{name_wo}.mat"
    mat_path = os.path.join(gt_dir, mat_name)

    if not os.path.exists(mat_path):
        return 0, None

    points = _read_points_from_mat(mat_path)
    count = int(points.shape[0])

    overlay_path = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        dm = density_from_points((w, h), points, sigma=15.0)
        # Draw points for clarity
        from utils import overlay_points_on_image
        vis = overlay_points_on_image(image, points, color=(0, 255, 0), radius=3)
        overlay_path = os.path.join(out_dir, f"overlay_{img_name}")
        vis.save(overlay_path)
    return count, overlay_path


def main():
    p = argparse.ArgumentParser(description="Batch GT-based crowd alert")
    p.add_argument("--images", required=True, help="Path to images directory")
    p.add_argument("--gt", required=True, help="Path to ground-truth directory containing GT_*.mat")
    p.add_argument("--to", required=True, help="Email address to alert")
    p.add_argument("--threshold", type=int, default=1, help="Minimum count to trigger alert")
    p.add_argument("--out", default="alerts_out", help="Directory to save visualization overlays")
    p.add_argument("--dry-run", action="store_true", help="Do not send email, just log")
    p.add_argument("--summary-csv", dest="summary_csv", default=None, help="Optional path to write a CSV summary of GT counts per image")
    # Optional SMTP creds (fallback to env if omitted)
    p.add_argument("--from-email", dest="from_email", default=None, help="SMTP From email (fallback: FROM_EMAIL env var)")
    p.add_argument("--smtp-server", dest="smtp_server", default=None, help="SMTP server (fallback: SMTP_SERVER env var)")
    p.add_argument("--smtp-port", dest="smtp_port", type=int, default=None, help="SMTP port (fallback: SMTP_PORT env var, default 587)")
    p.add_argument("--smtp-user", dest="smtp_user", default=None, help="SMTP username (fallback: SMTP_USER env var)")
    p.add_argument("--smtp-password", dest="smtp_password", default=None, help="SMTP password or app password (fallback: SMTP_PASSWORD env var)")
    args = p.parse_args()

    images = [f for f in os.listdir(args.images) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    images.sort()

    total_alerts = 0
    rows = []
    for fname in images:
        img_path = os.path.join(args.images, fname)
        count, overlay_path = process_image(img_path, args.gt, out_dir=args.out)
        print(f"{fname}: GT count = {count}")
        alerted = False
        if count >= args.threshold:
            total_alerts += 1
            subject = f"Crowd Alert: {count} people detected in {fname}"
            body = f"GT-based crowd count is {count} for image {fname}.\nImage path: {img_path}\nGT path: {os.path.join(args.gt, 'GT_' + os.path.splitext(fname)[0] + '.mat')}"
            if args.dry_run:
                print(f"[DRY RUN] Would send email to {args.to}: {subject}")
            else:
                send_email(
                    subject,
                    body,
                    to_email=args.to,
                    from_email=args.from_email,
                    smtp_server=args.smtp_server,
                    smtp_port=args.smtp_port,
                    smtp_user=args.smtp_user,
                    smtp_password=args.smtp_password,
                    attachment_path=overlay_path,
                )
                print(f"Alert email sent to {args.to}")
            alerted = True

        rows.append({
            'image': fname,
            'image_path': img_path,
            'gt_path': os.path.join(args.gt, 'GT_' + os.path.splitext(fname)[0] + '.mat'),
            'gt_count': count,
            'alerted': alerted,
            'overlay_path': overlay_path or ''
        })

    print(f"Done. Processed {len(images)} images. Alerts triggered: {total_alerts} (threshold={args.threshold}).")

    if args.summary_csv:
        os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True) if os.path.dirname(args.summary_csv) else None
        with open(args.summary_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image','image_path','gt_path','gt_count','alerted','overlay_path'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote summary CSV: {args.summary_csv}")


if __name__ == "__main__":
    main()
