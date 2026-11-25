"""按 S³OD 画布与分区绘制整图、切片推理和论文训练路径。"""

from html import escape
from pathlib import Path

from resvg_py import svg_to_bytes


DOCS = Path(__file__).resolve().parents[3] / "docs/reproduce/sahi"
INK, BLUE, YELLOW, GREEN = "#09557e", "#85d9f1", "#ffe8a2", "#bee09c"
ORANGE, PURPLE, AQUA = "#f6cba6", "#e6dff5", "#b9eeea"


def draw(sliced):
    title = "SAHI" if sliced else "整图推理"
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1430" viewBox="0 0 1400 1430" role="img" aria-labelledby="title">',
        f'<title id="title">{title} · VisDrone 水平框检测</title>',
        '<defs><marker id="end" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 1L9 5L0 9Z" fill="#738993"/></marker></defs>',
        '<g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif">',
    ]

    def box(x, y, width, height, fill, radius=6, edge="#8c9da6"):
        parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="{fill}" stroke="{edge}"/>')

    def text(x, y, value, size=16, weight=400, color="#263644", anchor="start"):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(value)}</text>')

    def wire(path, arrow=True, dashed=False):
        tip = ' marker-end="url(#end)"' if arrow else ""
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        parts.append(f'<path d="{path}" fill="none" stroke="#738993" stroke-width="1.5"{tip}{dash}/>')

    def node(x, y, width, height, heading, detail, fill):
        box(x, y, width, height, fill)
        center = x + width / 2
        text(center, y + height / 2 + (-4 if detail else 6), heading, 17, 700, INK, "middle")
        if detail:
            text(center, y + height / 2 + 19, detail, 14, 400, anchor="middle")

    def sheet(x, y, width, label):
        parts.append(f'<path d="M{x} {y}l34 -34h{width}l-34 34z M{x} {y}h{width}v15h-{width}z" fill="#fff" stroke="#4388bc"/>')
        text(x + width / 2, y + 12, label, 13, 700, INK, "middle")

    # 复用 S³OD 的外框、主干分界和细节区坐标，连线贯穿整图输入到输出。
    box(0, 0, 1400, 1430, "#fff", 0, "none")
    text(47, 51, title, 38, 700, "#0088d1")
    text(360, 50, "论文方法 · 切片推理与微调" if sliced else "论文基线 · Full Inference", 22, 600, INK)
    text(1351, 49, "VisDrone  /  水平框  /  本机 YOLO11s", 17, 600, "#428abb", "end")
    box(38, 65, 1324, 1322, "#ededed", 17, "#d7dadd")
    for boundary in ("M446 66V1386", "M446 239H1361", "M446 827H1361", "M39 353H445"):
        wire(boundary, False)
    text(50, 94, "Backbone", 24, 700, INK)
    text(51, 120, "同一检测器 · YOLO11s", 15, 500, INK)
    text(462, 96, "Head", 24, 700, INK)
    text(539, 95, "P3 / P4 / P5 → FPN + PAN → Detect", 16, 500, INK)
    for index, (y, width, stage) in enumerate(((307, 240, "RGB 输入"), (257, 213, "P3 · stride 8"),
                                                (207, 186, "P4 · stride 16"), (157, 159, "P5 · stride 32"))):
        x = 83 + 16 * index
        sheet(x, y, width, stage)
        if index:
            head_y = 202 - 28 * index
            wire(f"M{x + width + 17} {y - 17}H{410 + 24 * index}V{head_y}H550")
            node(552, head_y - 12, 132, 24, stage.split(" · ")[0], "", BLUE)
    for head_y in (174, 146, 118):
        wire(f"M684 {head_y}h33v-2h14")
    node(732, 112, 235, 82, "FPN + PAN", "对齐并融合三个尺度", GREEN)
    wire("M967 153H1000")
    node(1002, 112, 153, 82, "检测头", "分类 + 边框回归", BLUE)
    wire("M1155 153H1187")
    node(1189, 112, 151, 82, "视图内检测", "BBox + Class", "#cfdaeb")

    text(53, 382, "整图输入 → 视图 → 网络", 20, 700, INK)
    steps = [
        (405, "原始航拍图", "VisDrone val · 原图坐标", "#fff"),
        (539, "① 重叠切片" if sliced else "整图视图", "640×640 · 重叠 25%" if sliced else "每张原图只建一个视图", ORANGE),
        (673, "Letterbox", "单次输入张量 640×640", YELLOW),
        (807, "共享 YOLO11s", "卷积主干与多尺度特征", BLUE),
        (941, "FPN / PAN", "P3、P4、P5 双向融合", GREEN),
        (1075, "Detect", "回归水平框与类别分数", BLUE),
        (1209, "视图内 NMS", "IoU 0.7 · conf 0.001", PURPLE),
    ]
    for index, (y, heading, detail, fill) in enumerate(steps):
        node(106, y, 248, 74, heading, detail, fill)
        if index + 1 < len(steps):
            wire(f"M230 {y + 74}V{steps[index + 1][0] - 2}")
    if sliced:
        wire("M106 442H71V708H105", dashed=True)
        text(61, 520, "整图分支", 13, 700, INK)

    text(461, 270, "Details", 24, 700, INK)
    box(455, 283, 323, 522, YELLOW)
    text(470, 315, "① 输入视图", 22, 700, INK)
    box(481, 354, 269, 257, "#cfdaeb", 2)
    for x, y, width in ((511, 393, 12), (583, 428, 18), (673, 400, 14), (536, 542, 20), (644, 523, 12), (698, 575, 19)):
        box(x, y, width, 9, "#fff", 1, "#5b788a")
    if sliced:
        for x, y in ((484, 358), (597, 358), (484, 465), (597, 465)):
            parts.append(f'<rect x="{x}" y="{y}" width="150" height="142" fill="none" stroke="#a9532b" stroke-width="2" stroke-dasharray="6 3"/>')
    for y, line in ((656, "切片放大目标的相对像素面积。" if sliced else "整图缩放保留全部上下文。"),
                    (686, "每个切片独立检测，并保留原点。" if sliced else "远处的小目标缩放后可能只占几像素。"),
                    (740, "同时追加一次整图前向。" if sliced else "全图预测只调用一次检测器。")):
        text(470, y, line, 15)
    box(787, 283, 559, 522, "#e2e8ec")
    text(800, 315, "训练与监督", 22, 700, INK)
    node(816, 358, 500, 74, "图像与水平框标注", "基础检测器的有监督训练", "#fff")
    wire("M1066 432v32")
    node(816, 466, 500, 74, "③ 切片微调（论文 SF）" if sliced else "整图训练（论文 FI 权重）",
         "用裁片及对应裁剪框增广训练集" if sliced else "利用常规整图和原标注训练", ORANGE)
    wire("M1066 540v32")
    node(816, 574, 500, 74, "检测损失更新权重", "分类与边框监督由具体检测器提供", AQUA)
    text(813, 705, "本机配对直接加载同一份 YOLO11s 权重。", 17, 700, INK)
    text(813, 736, "权重来源和数据划分见复现卡。", 15)
    text(810, 776, "③ 本机切片推理实验未执行微调。" if sliced else "本机整图推理使用归档预训练权重。", 15)

    text(461, 857, "② 输出坐标与合并", 23, 700, INK)
    wire("M354 1246H414V951H478")
    node(480, 911, 265, 80, "视图坐标还原", "去除缩放和平移填充", BLUE)
    wire("M745 951H772")
    node(774, 911, 265, 80, "叠加切片原点" if sliced else "原图坐标", "加回切片左上角的偏移" if sliced else "直接使用还原框", AQUA)
    wire("M1039 951H1066")
    node(1068, 911, 265, 80, "按类别跨视图 NMS" if sliced else "整图预测框", "IoU 0.5 去重" if sliced else "保持视图内 NMS 结果", PURPLE)
    wire("M1200 991v54H906v41")
    node(667, 1088, 478, 94, "整图水平框输出", "保留前 500 框 · xywh / 类别 / 置信度", GREEN)
    wire("M906 1182v39")
    node(667, 1223, 478, 85, "统一 COCOeval", "AP、AP50、AP_small、AP50_small", YELLOW)
    text(905, 1350, "测速包含预处理、检测、坐标转换与合并，排除图像读盘。", 16, 500, INK, "middle")
    text(40, 1410, "论文依据  Fig. 2 / Table 1 · 作者 SAHI 0.11.36 · 本机实验仅切换推理流程", 15)
    return "\n".join([*parts, "</g></svg>", ""])


if __name__ == "__main__":
    DOCS.mkdir(parents=True, exist_ok=True)
    for name, sliced in (("full-inference-architecture", False), ("sahi-architecture", True)):
        svg = draw(sliced)
        (DOCS / f"{name}.svg").write_text(svg, encoding="utf-8")
        (DOCS / f"{name}.png").write_bytes(svg_to_bytes(svg_string=svg))
