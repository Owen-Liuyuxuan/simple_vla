#!/usr/bin/env python3
"""Stitch per-frame ``combine/*.jpg`` with ``occ_virtual/*.jpg`` and write an MP4.

Run from ``nuScenes/``::

    PYTHONPATH=./:$PYTHONPATH python tools/visualization/stitch_combine_occ_virtual_video.py \\
        --combine-dir output/debug_vis/combine \\
        --occ-virtual-dir output/debug_vis/occ_virtual \\
        -o output/debug_vis/combine_occ_virtual.mp4

The virtual rear view is resized to the combine strip height, then concatenated horizontally
(same idea as :func:`combine_frames_layout` in ``debug_test_single_gpu.py``).
"""
from __future__ import annotations

import argparse
import glob
import os
import os.path as osp
import re
import sys

import cv2
from tqdm import tqdm

_FRAME_RE = re.compile(r"^(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def _stem_index(path: str) -> int | None:
    base = osp.basename(path)
    m = _FRAME_RE.match(base)
    if not m:
        return None
    return int(m.group(1))


def resize_to_height_area(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / float(h)
    return cv2.resize(
        img, (int(round(w * scale)), target_h), interpolation=cv2.INTER_AREA
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--combine-dir",
        required=True,
        help="directory with combine frames (e.g. 0000.jpg)",
    )
    p.add_argument(
        "--occ-virtual-dir",
        required=True,
        help="directory with occ_virtual rear-camera frames",
    )
    p.add_argument(
        "-o",
        "--output",
        default="combine_occ_virtual.mp4",
        help="output MP4 path",
    )
    p.add_argument("--fps", type=int, default=12)
    p.add_argument(
        "--downsample",
        type=int,
        default=4,
        help="integer scale divisor before encoding (matches debug_test_single_gpu video)",
    )
    p.add_argument(
        "--layout",
        choices=("horizontal", "vertical"),
        default="horizontal",
        help="horizontal: resize occ_virtual to combine height then hconcat; "
        "vertical: resize occ_virtual to combine width then vconcat",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    combine_glob = glob.glob(osp.join(args.combine_dir, "*"))
    occ_glob = glob.glob(osp.join(args.occ_virtual_dir, "*"))

    def index_map(paths):
        out = {}
        for p in paths:
            idx = _stem_index(p)
            if idx is not None:
                out[idx] = p
        return out

    c_map = index_map(combine_glob)
    o_map = index_map(occ_glob)
    common = sorted(set(c_map.keys()) & set(o_map.keys()))
    if not common:
        print(
            "No matching indexed frames between combine_dir and occ_virtual_dir.",
            file=sys.stderr,
        )
        return 1

    out_dir = osp.dirname(osp.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    down = max(1, int(args.downsample))
    first = True
    size = None
    writer = None

    for idx in tqdm(common, desc="stitch+encode"):
        c_path = c_map[idx]
        v_path = o_map[idx]
        comb = cv2.imread(c_path)
        virt = cv2.imread(v_path)
        if comb is None or virt is None:
            continue
        ch = comb.shape[0]
        cw = comb.shape[1]
        if args.layout == "horizontal":
            if virt.shape[0] != ch:
                virt = resize_to_height_area(virt, ch)
            stitched = cv2.hconcat([comb, virt])
        else:
            if virt.shape[1] != cw:
                scale = cw / float(virt.shape[1])
                nh = int(round(virt.shape[0] * scale))
                virt = cv2.resize(virt, (cw, nh), interpolation=cv2.INTER_AREA)
            stitched = cv2.vconcat([comb, virt])

        h, w = stitched.shape[:2]
        stitched = cv2.resize(
            stitched,
            (w // down, h // down),
            interpolation=cv2.INTER_AREA,
        )
        h, w = stitched.shape[:2]
        if first:
            size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output, fourcc, float(args.fps), size)
            if not writer.isOpened():
                print(f"Failed to open VideoWriter for {args.output}", file=sys.stderr)
                return 1
            first = False
        elif (w, h) != size:
            stitched = cv2.resize(stitched, size, interpolation=cv2.INTER_AREA)
        writer.write(stitched)

    if writer is None:
        print("No frames written.", file=sys.stderr)
        return 1
    writer.release()
    print(f"Wrote {args.output} ({len(common)} frames @ {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
