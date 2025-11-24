#!/usr/bin/env python3
"""
rectify_no_crop.py

Rectify images for image_02 (camera 02) and image_03 (camera 03) using the
hard-coded calibration values (24-Nov-2025 block). No cropping is performed.
Outputs are written to recreated_rect/<seq>/image_0X/data/<original_filename>.
"""

from pathlib import Path
import numpy as np
import cv2

# -----------------------
# Hard-coded calibration (24-Nov-2025 block) for camera 02 and 03
# -----------------------
CALIB = {
    2: {  # camera 02 -> image_02
        "K": np.array([
            [1.021650e+03, 0.0,           6.935149e+02],
            [0.0,          9.934465e+02,  2.552404e+02],
            [0.0,          0.0,           1.0]
        ], dtype=np.float64),
        "D": np.array([-3.925910e-01, 2.937670e-01, 0.0, 0.0, 0.0], dtype=np.float64),
        "R_rect": np.array([
            [9.997076e-01, 9.697602e-03, -2.215189e-02],
            [-9.704203e-03, 9.999529e-01, -1.904950e-04],
            [2.214900e-02, 4.054057e-04, 9.997546e-01]
        ], dtype=np.float64),
        "P_rect": np.array([
            [9.693350e+02, 0.0,          7.178180e+02, 0.0],
            [0.0,          9.693350e+02, 2.551724e+02, 0.0],
            [0.0,          0.0,          1.0,         0.0]
        ], dtype=np.float64),
        "S_rect": (1392, 512)
    },
    3: {  # camera 03 -> image_03
        "K": np.array([
            [9.621979e+02, 0.0,          6.951952e+02],
            [0.0,          9.402923e+02, 2.552403e+02],
            [0.0,          0.0,          1.0]
        ], dtype=np.float64),
        "D": np.array([-4.414207e-01, 6.939901e-01, 0.0, 0.0, 0.0], dtype=np.float64),
        "R_rect": np.array([
            [9.998963e-01, -1.173631e-02, -8.348799e-03],
            [1.173880e-02, 9.999311e-01,  2.489748e-04],
            [8.345301e-03, -3.469539e-04, 9.999651e-01]
        ], dtype=np.float64),
        "P_rect": np.array([
            [9.693350e+02, 0.0,          7.178180e+02, -5.542457e+02],
            [0.0,          9.693350e+02, 2.551724e+02,  0.0],
            [0.0,          0.0,          1.0,          0.0]
        ], dtype=np.float64),
        "S_rect": (1392, 512)
    }
}

# -----------------------
# Rectify a single camera folder (no cropping)
# -----------------------
def rectify_folder_nocrop(calib_entry, in_dir: Path, out_dir: Path):
    """
    calib_entry: dictionary with keys 'K','D','R_rect','P_rect','S_rect'
    in_dir: Path to folder containing images (e.g., raw/seq_01/image_02/data)
    out_dir: destination folder (will be created)
    """
    image_paths = sorted([p for p in in_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if not image_paths:
        print(f"  - No images found in {in_dir}")
        return

    # Extract calibration matrices
    K = calib_entry["K"]
    D = calib_entry["D"]
    R = calib_entry["R_rect"]
    P = calib_entry["P_rect"]
    S = calib_entry["S_rect"]

    # target remap size (use S_rect)
    target_size = S if S is not None else None
    if target_size is None:
        # fallback to source image size of first image
        sample = cv2.imread(str(image_paths[0]))
        if sample is None:
            print(f"  - could not read sample image in {in_dir}")
            return
        h, w = sample.shape[:2]
        target_size = (w, h)

    # newK for remapping: left 3x3 of P_rect if present, otherwise K
    newK = P[:, :3] if P is not None else K.copy()

    # build remap once per folder
    try:
        map1, map2 = cv2.initUndistortRectifyMap(
            K.astype(np.float64), D.astype(np.float64),
            R.astype(np.float64), newK.astype(np.float64),
            target_size, cv2.CV_16SC2
        )
    except Exception as e:
        # try float maps if short integer maps fail
        map1, map2 = cv2.initUndistortRectifyMap(
            K.astype(np.float64), D.astype(np.float64),
            R.astype(np.float64), newK.astype(np.float64),
            target_size, cv2.CV_32FC1
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  - Rectifying {len(image_paths)} images from {in_dir} -> {out_dir} (size={target_size})")

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"    - Warning: could not read {p}, skipping")
            continue

        # If source isn't same dims as target_size, resize source to target_size before remap
        # (This is optional depending on expected source sizes; keeping for safety)
        if (img.shape[1], img.shape[0]) != target_size:
            src = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        else:
            src = img

        rect = cv2.remap(src, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.imwrite(str(out_dir / p.name), rect)


# -----------------------
# Batch: process all seq_* folders (image_02->cam2, image_03->cam3)
# -----------------------
def process_all(raw_root: Path, output_root: Path):
    CAMERA_MAP = {"image_02": 2, "image_03": 3}
    for seq in sorted(raw_root.glob("seq_*")):
        print(f"\nProcessing {seq.name}")
        for cam_folder, cam_id in CAMERA_MAP.items():
            in_dir = seq / cam_folder / "data"
            if not in_dir.exists():
                print(f"  - Skipping {cam_folder} (no data folder)")
                continue
            if cam_id not in CALIB:
                print(f"  - No calib for camera {cam_id}, skipping")
                continue
            out_dir = output_root / seq.name / cam_folder / "data"
            rectify_folder_nocrop(CALIB[cam_id], in_dir, out_dir)
    print("\nAll done.")


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    repo_root = Path.cwd()
    raw_root = repo_root / "raw"               # expects raw/seq_01, seq_02, ...
    output_root = repo_root / "recreated_rect" # will be created/filled
    print("Raw root:", raw_root)
    print("Output root:", output_root)
    process_all(raw_root, output_root)

    # sample uploaded file (local path) for your reference:
    # /mnt/data/7fa7f296-61be-409c-9df0-09a0fcc1e0c3.png
