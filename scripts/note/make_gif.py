"""Convert evaluation videos (MP4) into light GIFs for the note article.

Example:
    python.bat scripts/note/make_gif.py in.mp4 out.gif --width 480 --fps 12 --start 1 --seconds 8
"""

import argparse

import imageio.v3 as iio
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description="MP4 -> GIF for the note article.")
parser.add_argument("src", help="Input MP4.")
parser.add_argument("dst", help="Output GIF.")
parser.add_argument("--width", type=int, default=480, help="Output width [px].")
parser.add_argument("--fps", type=int, default=12, help="Output frame rate.")
parser.add_argument("--start", type=float, default=0.0, help="Start time [s].")
parser.add_argument("--seconds", type=float, default=8.0, help="Clip length [s].")
args = parser.parse_args()

meta = iio.immeta(args.src)
src_fps = float(meta.get("fps", 50))
frames = iio.imread(args.src)
step = max(1, round(src_fps / args.fps))
first = int(args.start * src_fps)
last = min(len(frames), first + int(args.seconds * src_fps))
height = round(frames.shape[1] * args.width / frames.shape[2])
clip = [np.asarray(Image.fromarray(f).resize((args.width, height), Image.LANCZOS)) for f in frames[first:last:step]]
iio.imwrite(args.dst, clip, duration=1000 * step / src_fps, loop=0)
print(f"[GIF] {args.dst}: {len(clip)} frames, {args.width}x{height}, {src_fps / step:.0f} fps")
