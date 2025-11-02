#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_video_anim.py
将 steps.json 的每步「Thought+Code」(左半屏) 与对应动画 step_XXX.mp4 (右半屏) 合成，
并顺序拼接为一个完整视频。左侧可随右侧动画时长自动滚动代码。

依赖：moviepy, pillow, pygments(可选), imageio-ffmpeg
"""
import os, json, argparse
from pathlib import Path
from typing import Tuple
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, VideoClip, CompositeVideoClip, concatenate_videoclips

# ---- 样式（与脚本①保持一致风格） ----
LEFT_BG = (20, 22, 26)
TEXT_COLOR = (235, 235, 235)
THOUGHT_COLOR = (180, 220, 255)
SEPARATOR_COLOR = (70, 70, 80)
PADDING = 28
GAP = 18
MONO_FALLBACKS = ["Menlo.ttc", "Consolas.ttf", "DejaVuSansMono.ttf", "JetBrainsMono-Regular.ttf"]

try:
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import ImageFormatter
    HAS_PYGMENTS = True
except Exception:
    HAS_PYGMENTS = False

def find_mono_font():
    for name in MONO_FALLBACKS:
        for d in ["/usr/share/fonts", "/Library/Fonts", str(Path.home()/"Library/Fonts"), "C:\\Windows\\Fonts"]:
            p = Path(d) / name
            if p.exists():
                return str(p)
    return None

def render_left_panel_pil(size: Tuple[int,int], thought: str, code: str, scroll_ratio: float = 0.0):
    """生成左半屏 PIL.Image；scroll_ratio∈[0,1] 表示代码区域从顶部滚到最底部的比例"""
    W,H = size
    img = Image.new("RGB", (W,H), LEFT_BG)
    draw = ImageDraw.Draw(img)

    font_path = find_mono_font()
    font_thought = ImageFont.truetype(font_path, size=28) if font_path else ImageFont.load_default()
    font_code   = ImageFont.truetype(font_path, size=24) if font_path else ImageFont.load_default()
    font_label  = ImageFont.truetype(font_path, size=22) if font_path else ImageFont.load_default()

    x, y = PADDING, PADDING

    # Thought
    draw.text((x,y), "Thought", fill=THOUGHT_COLOR, font=font_label)
    y += 8 + font_label.size

    # 简单折行
    def wrap(text, font, maxw):
        out=[]; 
        for para in text.split("\n"):
            if not para: out.append(""); continue
            buf=""
            for ch in para:
                test = buf + ch
                w,_ = draw.textsize(test, font=font)
                if w<=maxw: buf=test
                else: out.append(buf); buf=ch
            out.append(buf)
        return out

    maxw = W - 2*PADDING
    for line in wrap(thought, font_thought, maxw):
        draw.text((x,y), line, fill=TEXT_COLOR, font=font_thought)
        y += font_thought.size + 6

    y += GAP
    draw.line((PADDING, y, W-PADDING, y), fill=SEPARATOR_COLOR, width=2)
    y += GAP

    draw.text((x,y), "Code (Python)", fill=(180,255,200), font=font_label)
    y += 8 + font_label.size

    code_top_y = y
    code_box_h = H - PADDING - y
    code_box = Image.new("RGB", (maxw, code_box_h), (16,16,18))

    # 渲染代码图
    if HAS_PYGMENTS:
        from io import BytesIO
        formatter = ImageFormatter(
            font_name="DejaVu Sans Mono", image_format="PNG",
            line_numbers=False, line_pad=2, font_size=20, style="native"
        )
        bio = BytesIO()
        bio.write(highlight(code, PythonLexer(), formatter))
        bio.seek(0)
        code_img = Image.open(bio).convert("RGB")
        # 宽度适配
        if code_img.width > 0:
            ratio = maxw / code_img.width
            code_img = code_img.resize((maxw, max(1,int(code_img.height*ratio))), Image.BICUBIC)
    else:
        code_lines = code.split("\n")
        est_h = max(10, (font_code.size+6)*len(code_lines) + 10)
        code_img = Image.new("RGB", (maxw, est_h), (16,16,18))
        d2 = ImageDraw.Draw(code_img); yy=6
        for ln in code_lines:
            d2.text((8,yy), ln if ln else " ", fill=(220,220,220), font=font_code)
            yy += font_code.size + 6

    # 滚动
    max_scroll = max(0, code_img.height - code_box_h)
    s = int(max_scroll * min(max(scroll_ratio,0.0),1.0))
    crop = code_img.crop((0, s, maxw, min(s+code_box_h, code_img.height)))
    code_box.paste(crop, (0,0))
    img.paste(code_box, (x, code_top_y))

    # 滚动条
    if max_scroll > 0:
        bar_h = max(30, int(code_box_h * (code_box_h / code_img.height)))
        bar_y = code_top_y + int((code_box_h - bar_h) * (s / max_scroll))
        draw.rectangle((W - PADDING - 6, bar_y, W - PADDING - 2, bar_y + bar_h), fill=(90,90,100))
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=str, required=True)
    ap.add_argument("--videos_dir", type=str, default="renders_anim")
    ap.add_argument("--out", type=str, default="video_anim.mp4")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--scroll_code", action="store_true", help="左侧代码随时长平滑滚动")
    ap.add_argument("--fps", type=int, default=30, help="输出视频帧率（不改变右侧源动画的时间，仅重采样）")
    args = ap.parse_args()

    steps = json.loads(Path(args.steps).read_text(encoding="utf-8"))
    half_w = args.width // 2

    clips = []
    for i, step in enumerate(steps, start=1):
        thought = step.get("thought","")
        code = step.get("code","")

        # 右侧动画
        vid_path = Path(step.get("video_path","")) if step.get("video_path") else Path(args.videos_dir) / f"step_{i:03d}.mp4"
        if not vid_path.exists():
            raise FileNotFoundError(f"找不到动画：{vid_path}")
        right = VideoFileClip(str(vid_path)).resize(width=half_w).set_position((half_w,0))

        # 左侧：根据时间 t 生成帧（静态 + 可选滚动）
        dur = right.duration
        def make_left_frame(t):
            # 按 t 映射滚动比例
            scroll_ratio = (t / max(dur, 0.001)) if args.scroll_code else 0.0
            pil = render_left_panel_pil((half_w, args.height), thought, code, scroll_ratio)
            return pil

        left_clip = VideoClip(lambda t: make_left_frame(t), duration=dur).set_position((0,0))
        left_clip = left_clip.set_duration(dur).resize((half_w, args.height))

        comp = CompositeVideoClip([left_clip, right], size=(args.width, args.height)).set_duration(dur)
        clips.append(comp)

    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(args.out, fps=args.fps, codec="libx264", audio=False, bitrate="8000k")
    print(f"[OK] 写出合成视频：{args.out}")

if __name__ == "__main__":
    main()
