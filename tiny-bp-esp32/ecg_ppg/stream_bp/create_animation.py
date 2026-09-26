"""Render case_236_trace.json as a self-contained animated SVG."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
TRACE = json.loads((HERE / "case_236_trace.json").read_text())
OUT = HERE / "case_236_stream_fit.svg"

W, H = 1200, 790
LEFT, RIGHT = 92, 1144
PANELS = (("SBP", 150, 383, 55, 195, 0), ("DBP", 468, 700, 30, 115, 1))
COLORS = {"reference_bp_mmhg": "#e9f2ff", "ppg_only_bp_mmhg": "#f5b64d", "ppg_pat_rr_bp_mmhg": "#4fe0c5"}
LABELS = {"reference_bp_mmhg": "ART reference", "ppg_only_bp_mmhg": "PPG", "ppg_pat_rr_bp_mmhg": "PPG + ECG timing"}


def xcoord(seconds):
    return LEFT + (RIGHT - LEFT) * seconds / 1800


def ycoord(value, top, bottom, low, high):
    return bottom - (value - low) * (bottom - top) / (high - low)


def path(points):
    return " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))


svg = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Animated blood pressure comparison for VitalDB case 236">
<title>Streaming blood pressure fit, VitalDB case 236</title>
<desc>Two charts compare invasive arterial pressure with PPG-only and PPG plus ECG timing estimates over 30 minutes after calibration. Models track direction but underestimate the rise. Values are 10-second window summaries, not beat-by-beat pressure.</desc>
<style>
text {{ font-family: Arial, 'Microsoft YaHei', sans-serif; fill: #eaf3ff; }}
.muted {{ fill: #a9bdd3; }} .grid {{ stroke: #32475d; stroke-width: 1; }}
.plot {{ fill: none; stroke-linecap: round; stroke-linejoin: round; }}
</style>
<rect width="1200" height="790" fill="#0c1726"/>
<text x="56" y="53" font-size="27" font-weight="700">连续血压估计：一例真实波形回放</text>
<text x="56" y="82" class="muted" font-size="15">VitalDB 病例 236 · 校准后 30 分钟 · 175 个有效 10 秒窗口 · 单位 mmHg</text>
<rect x="800" y="25" width="344" height="70" rx="10" fill="#172a3d"/>
<text x="819" y="50" font-size="14">融合模型 MAE</text>
<text x="819" y="78" font-size="21" font-weight="700" fill="#4fe0c5">SBP 26.51  /  DBP 13.66</text>
<defs><clipPath id="reveal"><rect x="{LEFT}" y="112" width="0" height="612"><animate attributeName="width" from="0" to="{RIGHT-LEFT}" dur="20s" repeatCount="indefinite"/></rect></clipPath></defs>
''']

for title, top, bottom, low, high, channel in PANELS:
    svg.append(f'<rect x="56" y="{top-36}" width="1088" height="{bottom-top+65}" rx="12" fill="#132338"/>')
    svg.append(f'<text x="{LEFT}" y="{top-12}" font-size="18" font-weight="700">{title}  收缩压</text>' if channel == 0 else f'<text x="{LEFT}" y="{top-12}" font-size="18" font-weight="700">{title}  舒张压</text>')
    for tick in (range(60, 196, 20) if channel == 0 else range(40, 116, 20)):
        y = ycoord(tick, top, bottom, low, high)
        svg.append(f'<line class="grid" x1="{LEFT}" x2="{RIGHT}" y1="{y:.1f}" y2="{y:.1f}"/><text x="{LEFT-12}" y="{y+5:.1f}" text-anchor="end" class="muted" font-size="13">{tick}</text>')
    for minute in (0, 5, 10, 15, 20, 25, 30):
        x = xcoord(minute * 60)
        svg.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{bottom}" opacity=".45"/>')
        if channel == 1:
            svg.append(f'<text x="{x:.1f}" y="{bottom+24}" class="muted" font-size="13" text-anchor="middle">{minute}</text>')
    baseline = TRACE["calibration_bp_mmhg"][channel]
    y = ycoord(baseline, top, bottom, low, high)
    svg.append(f'<line x1="{LEFT}" x2="{RIGHT}" y1="{y:.1f}" y2="{y:.1f}" stroke="#bd8bf4" stroke-width="1.5" stroke-dasharray="7 6" opacity=".75"/>')
    for key in COLORS:
        points = [(xcoord(t), ycoord(bp[channel], top, bottom, low, high)) for t, bp in zip(TRACE["elapsed_s"], TRACE[key])]
        d = path([(LEFT, ycoord(baseline, top, bottom, low, high))] + points)
        color = COLORS[key]
        width = 3.5 if key == "reference_bp_mmhg" else 2.6
        svg.append(f'<path class="plot" d="{d}" stroke="{color}" stroke-width="{width}" opacity=".13"/>')
        svg.append(f'<path class="plot" d="{d}" stroke="{color}" stroke-width="{width}" clip-path="url(#reveal)"/>')

svg.append('''<text x="600" y="742" text-anchor="middle" class="muted" font-size="14">校准后时间（分钟）</text>
<line x1="92" y1="764" x2="117" y2="764" stroke="#e9f2ff" stroke-width="4"/><text x="125" y="769" font-size="13">有创动脉压参考</text>
<line x1="305" y1="764" x2="330" y2="764" stroke="#f5b64d" stroke-width="4"/><text x="338" y="769" font-size="13">仅 PPG</text>
<line x1="455" y1="764" x2="480" y2="764" stroke="#4fe0c5" stroke-width="4"/><text x="488" y="769" font-size="13">PPG + ECG 时序</text>
<line x1="685" y1="764" x2="710" y2="764" stroke="#bd8bf4" stroke-width="2" stroke-dasharray="6 4"/><text x="718" y="769" font-size="13">校准值不变</text>
<text x="1144" y="769" text-anchor="end" class="muted" font-size="12">动画每 20 秒循环</text>
</svg>''')
OUT.write_text("\n".join(svg), encoding="utf-8")
print(OUT)
