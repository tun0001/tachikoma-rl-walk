"""Stitch evaluation clips and title cards into one captioned video (real-time playback).

The storyboard is a JSON list of segments:
    {"card": ["Title", "line", ...], "seconds": 4}
    {"clip": "docs/note/videos/01_v1_flat.mp4", "start": 1, "seconds": 8, "caption": "...", "sub": "..."}

Example:
    python.bat scripts/note/make_final_video.py docs/submission/storyboard.json docs/submission/tachikoma_final.mp4
"""

import argparse
import json

import imageio.v2 as imageio
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description="Captioned video from a JSON storyboard.")
parser.add_argument("storyboard", help="Storyboard JSON.")
parser.add_argument("dst", help="Output MP4.")
parser.add_argument("--fps", type=int, default=25, help="Output frame rate (clips are subsampled, not sped up).")
parser.add_argument("--font", default="C:/Windows/Fonts/YuGothB.ttc", help="Font with Japanese glyphs.")
args = parser.parse_args()

W, H = 1280, 720
font_title = ImageFont.truetype(args.font, 52)
font_line = ImageFont.truetype(args.font, 34)
font_caption = ImageFont.truetype(args.font, 36)
font_sub = ImageFont.truetype(args.font, 26)
font_badge = ImageFont.truetype(args.font, 22)


def card(lines: list[str]) -> np.ndarray:
    img = Image.new("RGB", (W, H), (18, 24, 38))
    draw = ImageDraw.Draw(img)
    fonts = [font_title] + [font_line] * (len(lines) - 1)
    heights = [f.getbbox(t or " ")[3] + 28 for f, t in zip(fonts, lines)]
    y = (H - sum(heights)) // 2
    for f, t, h in zip(fonts, lines, heights):
        draw.text((W // 2, y), t, font=f, fill=(255, 255, 255) if f is font_title else (190, 205, 230), anchor="ma")
        y += h
    return np.asarray(img)


def captioned(frame: np.ndarray, caption: str, sub: str) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    bar = 130 if sub else 80
    draw.rectangle([0, H - bar, W, H], fill=(0, 0, 0, 170))
    draw.text((32, H - bar + 18), caption, font=font_caption, fill=(255, 255, 255, 255))
    if sub:
        draw.text((32, H - bar + 72), sub, font=font_sub, fill=(200, 210, 230, 255))
    draw.rounded_rectangle([W - 150, 16, W - 16, 52], radius=8, fill=(0, 0, 0, 150))
    draw.text((W - 83, 34), "等倍速", font=font_badge, fill=(255, 255, 255, 255), anchor="mm")
    return np.asarray(Image.alpha_composite(img, overlay).convert("RGB"))


with open(args.storyboard, encoding="utf-8") as f:
    storyboard = json.load(f)

# stream frame by frame: a whole 2-3 min video at 720p would not fit in memory
writer = imageio.get_writer(args.dst, fps=args.fps, codec="libx264", quality=7, macro_block_size=8)
count = 0
for seg in storyboard:
    if "card" in seg:
        frame = card(seg["card"])
        for _ in range(round(seg["seconds"] * args.fps)):
            writer.append_data(frame)
        count += round(seg["seconds"] * args.fps)
        continue
    src_fps = float(iio.immeta(seg["clip"]).get("fps", 50))
    first = int(seg.get("start", 0) * src_fps)
    last = first + int(seg["seconds"] * src_fps)
    step = src_fps / args.fps
    next_pick = float(first)
    for i, frame in enumerate(iio.imiter(seg["clip"])):
        if i >= last:
            break
        if i < round(next_pick):
            continue
        next_pick += step
        if frame.shape[:2] != (H, W):
            frame = np.asarray(Image.fromarray(frame).resize((W, H), Image.LANCZOS))
        writer.append_data(captioned(frame, seg.get("caption", ""), seg.get("sub", "")))
        count += 1
writer.close()
print(f"[VIDEO] {args.dst}: {count / args.fps:.1f} s, {count} frames, {args.fps} fps")
