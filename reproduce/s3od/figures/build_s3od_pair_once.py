from html import escape
from pathlib import Path
import re
import subprocess


folder = Path("docs/reproduce/s3od")
source = subprocess.check_output(["git", "show", "4b56b89:docs/paper/figures/s3od-architecture.svg"]).decode("utf-8")
definitions = source[source.index("<defs>"):source.index("</style>") + len("</style>")]
layers = source.split('<g transform="translate(0 510)">\n', 1)[1].rsplit("</g>", 1)[0]
original_cards = re.findall(r'<text [^>]+>(.*?)</text>', layers[layers.index('<rect x="787" y="381"'):layers.index('<rect x="787" y="662"')])
assert len(original_cards) == 27

baseline_cards = [
    "固定阈值", "教师输出 → 伪标签", "教师框 + 分数", "所有尺度共用门槛",
    "按固定阈值筛选", "RPN / 分类取 0.9", "保留高分伪框", "坐标转到强增强视图", "同一阈值容易漏掉小框",
    "IoU 分配", "学生 RPN → 正锚框", "锚框与目标框", "比较两框的重叠程度",
    "MaxIoU 分配", "含低质量匹配补充", "常规采样与损失", "正样本按原规则学习", "小框位置偏差影响 IoU",
    "负样本采样", "学生 RoI → 分类损失", "学生候选区域", "与伪框计算匹配度",
    "按 IoU 分配背景", "RoI 负样本 IoU < 0.5", "随机采样候选", "背景标签进入分类损失", "漏标的小目标可能被误罚",
]
palette = ["#9e4d1b", "#266294", "#146d72"]


def badge(x, y, number, label=""):
    color = palette[number-1]
    width = 25 if not label else 160
    background = f'<rect x="{x}" y="{y}" width="{width}" height="24" rx="5" fill="{color}"/>'
    text = f'{number}　{label}' if label else str(number)
    return background + f'<text x="{x+width/2}" y="{y+17}" font-size="14" font-weight="700" text-anchor="middle" style="fill:white">{text}</text>'


for baseline in (True, False):
    stem = "ssod-baseline-architecture" if baseline else "s3od-architecture"
    name = "SSOD baseline" if baseline else "S³OD"
    drawing = layers.replace(">S³OD</text>", f'>{name}</text>', 1)
    drawing = drawing.replace('x="171" y="50"', 'x="360" y="50"')
    drawing = drawing.replace("检测器逐层结构与训练模块展开", "基线模型 · 训练监督" if baseline else "论文模型 · 训练监督")
    start = drawing.index('<rect x="787" y="381"')
    end = drawing.index('<rect x="787" y="662"')
    cards = drawing[start:end]
    if baseline:
        labels = iter(baseline_cards)
        cards = re.sub(r'(<text [^>]+>).*?(</text>)', lambda match: match[1]+escape(next(labels))+match[2], cards)
        for color in ["#f5cda9", "#9dc7ed", "#aaebf3"]:
            cards = cards.replace(color, "#dde3e7")
    for x, number in [(933, 1), (1124, 2), (1314, 3)]:
        cards += badge(x, 389, number)
    drawing = drawing[:start]+cards+drawing[end:]
    labels = ["固定阈值筛伪框", "IoU 正锚框分配", "常规负样本"] if baseline else ["SAT 尺度阈值", "SLA 锚框分配", "TNL 负学习"]
    for connector in ["M1185 1293v16", "M1009 916v11", "M1058 1152v27h7"]:
        drawing += f'<path class="wire" d="{connector}" stroke-dasharray="3 2"/>'
    drawing += badge(1105, 1309, 1, labels[0])
    drawing += badge(929, 927, 2, labels[1])
    drawing += badge(906, 1128, 3, labels[2])
    drawing = drawing.replace("接入点：教师输出", "教师预测后筛选伪框").replace("接入点：锚框分配 / 损失", "学生 RPN 正锚框与损失").replace("接入点：RoI 分类损失", "学生 RoI 负类损失")
    score = "36.0" if baseline else "41.9"
    drawing = re.sub(r'依据：S³OD 预印本.*?</text>', f'作者报告 · DOTA-v1.5 · 1% 单划分 · mAP@0.50 = {score}（预印本表 III）。结构依据公开配置 d7806b7。编号 1–3 对应训练位置。</text>', drawing)
    contents = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1430" viewBox="0 0 1400 1430" role="img" aria-labelledby="title desc">
<title id="title">{name} 逐层结构与三处训练监督</title>
<desc id="desc">1 标记教师输出的伪框筛选，2 标记学生 RPN 正样本分配，3 标记学生 RoI 分类负样本处理。对应编号在流程与模块细节中各出现一次。</desc>
{definitions}
{drawing}
</svg>
"""
    (folder / f"{stem}.svg").write_text(contents, encoding="utf-8")
