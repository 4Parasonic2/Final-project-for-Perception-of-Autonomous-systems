#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import cv2

CALIB = {
    2: {
        "K": np.array([
            [1.021650e+03, 0.0,           6.935149e+02],
            [0.0,          9.934465e+02,  2.552404e+02],
            [0.0,          0.0,           1.0]
        ]),
        "D": np.array([-3.925910e-01, 2.937670e-01, 0., 0., 0.]),
        "R_rect": np.array([
            [9.997076e-01, 9.697602e-03, -2.215189e-02],
            [-9.704203e-03, 9.999529e-01, -1.904950e-04],
            [2.214900e-02, 4.054057e-04, 9.997546e-01]
        ]),
        "P_rect": np.array([
            [9.693350e+02, 0.,           7.178180e+02, 0.],
            [0.,           9.693350e+02, 2.551724e+02, 0.],
            [0.,           0.,           1.,           0.]
        ]),
        "S_rect": (1392, 512)
    },
    3: {
        "K": np.array([
            [9.621979e+02, 0.,          6.951952e+02],
            [0.,          9.402923e+02, 2.552403e+02],
            [0.,          0.,           1.]
        ]),
        "D": np.array([-4.414207e-01, 6.939901e-01, 0., 0., 0.]),
        "R_rect": np.array([
            [9.998963e-01, -1.173631e-02, -8.348799e-03],
            [1.173880e-02,  9.999311e-01,  2.489748e-04],
            [8.345301e-03, -3.469539e-04,  9.999651e-01]
        ]),
        "P_rect": np.array([
            [9.693350e+02, 0.,          7.178180e+02, -5.542457e+02],
            [0.,           9.693350e+02, 2.551724e+02,  0.],
            [0.,           0.,           1.,           0.]
        ]),
        "S_rect": (1392, 512)
    }
}

def rectify_folder(calib, in_dir: Path, out_dir: Path):
    imgs = sorted([p for p in in_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if not imgs:
        print(f"No images in {in_dir}")
        return

    K, D, R, P = calib["K"], calib["D"], calib["R_rect"], calib["P_rect"]
    target = calib["S_rect"]

    if target is None:
        sample = cv2.imread(str(imgs[0]))
        if sample is None:
            print(f"Unreadable sample in {in_dir}")
            return
        h, w = sample.shape[:2]
        target = (w, h)

    newK = P[:, :3]

    try:
        map1, map2 = cv2.initUndistortRectifyMap(K, D, R, newK, target, cv2.CV_16SC2)
    except:
        map1, map2 = cv2.initUndistortRectifyMap(K, D, R, newK, target, cv2.CV_32FC1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(imgs)} images -> {out_dir}")

    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue

        if (img.shape[1], img.shape[0]) != target:
            img = cv2.resize(img, target)

        rect = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite(str(out_dir / p.name), rect)

def process_all(raw_root: Path, out_root: Path):
    cams = {"image_02": 2, "image_03": 3}
    for seq in sorted(raw_root.glob("seq_*")):
        print("\n", seq.name)
        for folder, cam_id in cams.items():
            src = seq / folder / "data"
            if not src.exists() or cam_id not in CALIB:
                continue
            dst = out_root / seq.name / folder / "data"
            rectify_folder(CALIB[cam_id], src, dst)
    print("Done.")

if __name__ == "__main__":
    root = Path.cwd()
    raw = root / "raw"
    out = root / "recreated_rect"
    print("Raw:", raw)
    print("Out:", out)
    process_all(raw, out)
