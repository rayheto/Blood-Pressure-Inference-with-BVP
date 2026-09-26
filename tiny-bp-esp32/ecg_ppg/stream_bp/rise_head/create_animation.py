"""Create an animated, trailing-average comparison from the raw replay trace."""

import json
from pathlib import Path
from xml.sax.saxutils import escape


HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "case_236_trace.json").read_text())
out = HERE / "case_236_rise_fit.svg"
times = data["elapsed_s"]
width, height = 1200, 790
left, right = 90, 1150
series = (
    ("reference_bp_mmhg", "有创动脉压参考", "#f1f6ff", 3.5),
    ("ppg_pat_rr_bp_mmhg", "原融合模型", "#f6b955", 2.5),
    ("uniform_bp_mmhg", "普通校正头", "#51e2c2", 2.7),
    ("rise_head_bp_mmhg", "升压幅度校正头", "#ba99ff", 2.7),
)


def avg_60(values):
    return [[sum(values[j][axis] for j, past in enumerate(times) if 0 <= t - past < 60) /
             sum(1 for past in times if 0 <= t - past < 60)
             for axis in (0, 1)] for t in times]


def line(values, axis, top, bottom, low, high):
    coords = [(left, bottom - (data["calibration_bp_mmhg"][axis] - low) * (bottom - top) / (high - low))]
    coords += [(left + (right - left) * t / 1800,
                bottom - (bp[axis] - low) * (bottom - top) / (high - low))
               for t, bp in zip(times, avg_60(values))]
    return " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(coords))


svg = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Animated comparison of base and dynamically corrected blood pressure estimates">
<title>流式血压升压幅度校正：VitalDB 病例 236</title>
<desc>校准后30分钟的175个有效窗口，四条曲线使用相同的60秒后向平均。升压幅度校正头改善此病例，但需在未被查看过的新病例上验证。</desc>
<style>text {{ font-family: Arial, 'Microsoft YaHei', sans-serif; fill:#ebf4ff }} .muted {{ fill:#abc1d6 }} .grid {{ stroke:#33495d;stroke-width:1 }}</style>
<rect width="1200" height="790" fill="#0c1726"/>
<text x="55" y="50" font-size="27" font-weight="700">升压幅度校正：流式血压拟合对比</text>
<text x="55" y="80" font-size="15" class="muted">病例 236 · 校准后 30 分钟 · 175 个窗口 · 相同的 60 秒后向平滑</text>
<rect x="810" y="20" width="340" height="78" rx="10" fill="#172a3d"/>
<text x="830" y="45" font-size="14">本例原始 MAE（SBP / DBP）</text>
<text x="830" y="70" font-size="17">原模型 26.51 / 13.66</text>
<text x="830" y="91" font-size="17">升压校正 8.07 / 5.46</text>
<defs><clipPath id="reveal"><rect x="{left}" y="112" width="0" height="612"><animate attributeName="width" from="0" to="{right-left}" dur="20s" repeatCount="indefinite"/></rect></clipPath></defs>''']

for axis, (label, top, bottom, low, high, ticks) in enumerate((
    ("SBP 收缩压", 151, 383, 55, 195, range(60, 196, 20)),
    ("DBP 舒张压", 468, 700, 30, 115, range(40, 116, 20)),
)):
    svg.append(f'<rect x="55" y="{top-36}" width="1095" height="{bottom-top+65}" rx="12" fill="#132338"/>')
    svg.append(f'<text x="{left}" y="{top-12}" font-size="18" font-weight="700">{label}</text>')
    for tick in ticks:
        y = bottom - (tick - low) * (bottom - top) / (high - low)
        svg.append(f'<line class="grid" x1="{left}" x2="{right}" y1="{y:.1f}" y2="{y:.1f}"/><text x="{left-10}" y="{y+5:.1f}" text-anchor="end" class="muted" font-size="13">{tick}</text>')
    for minute in range(0, 31, 5):
        x = left + (right - left) * minute / 30
        svg.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{bottom}" opacity=".45"/>')
        if axis == 1:
            svg.append(f'<text x="{x:.1f}" y="{bottom+23}" text-anchor="middle" class="muted" font-size="13">{minute}</text>')
    for key, _, color, stroke in series:
        path = line(data[key], axis, top, bottom, low, high)
        svg.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linejoin="round" opacity=".12"/>')
        svg.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linejoin="round" clip-path="url(#reveal)"/>')

svg.append('<text x="600" y="742" text-anchor="middle" class="muted" font-size="14">校准后时间（分钟）</text>')
for i, (_, name, color, _) in enumerate(series):
    x = (90, 325, 550, 765)[i]
    svg.append(f'<line x1="{x}" x2="{x+25}" y1="765" y2="765" stroke="{color}" stroke-width="4"/><text x="{x+33}" y="770" font-size="13">{escape(name)}</text>')
svg.append('<text x="1146" y="770" text-anchor="end" class="muted" font-size="12">20 秒循环</text></svg>')
out.write_text("\n".join(svg), encoding="utf-8")
print(out)
