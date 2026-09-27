"""Export these SVG reveal animations to H.264 MP4 using CairoSVG and FFmpeg."""

import argparse
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cairosvg
import imageio_ffmpeg


WIDTH, HEIGHT = 1200, 790
LEFT, REVEAL_WIDTH = 90, 1060
DURATION_SECONDS, FPS = 20, 24
REGULAR_FONT = Path(r"C:\Windows\Fonts\msyh.ttc")
BOLD_FONT = Path(r"C:\Windows\Fonts\msyhbd.ttc")


def text_specs(source):
    root = ET.fromstring(source.read_text(encoding="utf-8"))
    for element in root.iter("{http://www.w3.org/2000/svg}text"):
        yield (float(element.attrib["x"]), float(element.attrib["y"]),
               int(element.attrib.get("font-size", "16")),
               element.attrib.get("font-weight") == "700",
               element.attrib.get("text-anchor", "start"),
               element.attrib.get("class") == "muted", element.text or "")


def static_svg(source, width):
    content = source.read_text(encoding="utf-8")
    # Cairo's Windows font fallback drops CJK glyphs. PIL renders the original
    # SVG text with Microsoft YaHei after Cairo rasterizes the geometry.
    content = re.sub(r"<text\b[^>]*>.*?</text>", "", content, flags=re.DOTALL)
    content, count = re.subn(r'<animate attributeName="width"[^>]*/>', "", content)
    if count != 1:
        raise ValueError("Expected one width animation")
    content, count = re.subn(r'(<clipPath id="reveal"><rect x="90" y="112" width=")0',
                             lambda match: match.group(1) + str(width), content)
    if count != 1:
        raise ValueError("Expected one reveal clip")
    return content


def screenshot(svg, png, specs):
    cairosvg.svg2png(bytestring=svg.read_bytes(), write_to=str(png))
    image = Image.open(png).convert("RGB")
    if image.size != (WIDTH, HEIGHT):
        raise ValueError(f"Unexpected screenshot dimensions: {image.size}")
    draw = ImageDraw.Draw(image)
    for x, y, size, bold, align, muted, content in specs:
        font = ImageFont.truetype(str(BOLD_FONT if bold else REGULAR_FONT), size)
        anchor = {"start": "ls", "middle": "ms", "end": "rs"}[align]
        draw.text((x, y), content, fill="#abc1d6" if muted else "#ebf4ff",
                  font=font, anchor=anchor)
    return np.asarray(image)


def export(source, output):
    specs = list(text_specs(source))
    with tempfile.TemporaryDirectory(prefix="bp-svg-video-") as temp:
        temp = Path(temp)
        background_svg, foreground_svg = temp / "background.svg", temp / "foreground.svg"
        background_svg.write_text(static_svg(source, 0), encoding="utf-8")
        foreground_svg.write_text(static_svg(source, REVEAL_WIDTH), encoding="utf-8")
        background = screenshot(background_svg, temp / "background.png", specs)
        foreground = screenshot(foreground_svg, temp / "foreground.png", specs)
        if np.array_equal(background, foreground):
            raise RuntimeError("Rendered start and end frames are identical")
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
                   "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
                   "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            for frame_index in range(FPS * DURATION_SECONDS):
                visible = round(REVEAL_WIDTH * frame_index / (FPS * DURATION_SECONDS - 1))
                frame = background.copy()
                if visible:
                    frame[:, LEFT:LEFT + visible] = foreground[:, LEFT:LEFT + visible]
                encoder.stdin.write(frame.tobytes())
        except Exception:
            encoder.kill()
            raise
        finally:
            encoder.stdin.close()
        error = encoder.stderr.read().decode(errors="replace")
        if encoder.wait() != 0:
            raise RuntimeError(f"FFmpeg failed: {error}")
    print(f"{output} ({output.stat().st_size} bytes)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("svg", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.svg:
        export(path.resolve(), path.with_suffix(".mp4").resolve())


if __name__ == "__main__":
    main()
