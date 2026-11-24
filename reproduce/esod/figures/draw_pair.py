"""用 S³OD 参考图的坐标、标记和配色绘制两张可直接预览的 SVG。"""

from html import escape
from pathlib import Path


DOCS = Path(__file__).resolve().parents[3] / "docs/reproduce"
BLUE, YELLOW, GREEN = "#85d9f1", "#ffe8a2", "#bee09c"
ORANGE, PURPLE, AQUA = "#f6cba6", "#e6dff5", "#b9eeea"
INK, BORDER = "#09557e", "#8c9da6"
ACCENTS = ("#9e4d1b", "#266294", "#146d72")


def draw(esod):
    name = "ESOD" if esod else "YOLOv5m"
    # 读取仓库中已发布的参考图样式，保留相同的箭头、字形与外框。
    reference = (DOCS / "s3od/s3od-architecture.svg").read_text(encoding="utf-8")
    style = reference[reference.index("<defs>"):reference.index("</style>") + len("</style>")]
    street = '''<pattern id="street" patternUnits="userSpaceOnUse" width="96" height="66">
      <rect width="96" height="66" fill="#afbaa9"/><path d="M0 28h96m-67-28v66" stroke="#7e8d8b" stroke-width="14"/>
      <path d="M0 28h96" stroke="#efe4bd" stroke-dasharray="7 4"/><g fill="#d6cbb9" stroke="#8d9189">
      <path d="M4 4h14v13H4zM43 5h39v12H43zM45 44h18v17H45z"/></g>
      <path d="M4 25h8m31 7h8m17-7h8" stroke="#f3ecdc" stroke-width="3"/></pattern>'''
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1430" viewBox="0 0 1400 1430" role="img" aria-labelledby="title desc">',
        f'<title id="title">{name} · VisDrone 水平框检测全流程</title>',
        '<desc id="desc">上方是整图数据流。左列主干的三个侧输出连入下方双向金字塔，再由右侧检测头输出整图水平框。编号对应中部算子展开。</desc>',
        style.replace("</defs>", street + "</defs>"),
    ]

    def box(x, y, width, height, fill, radius=4, edge=BORDER):
        parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="{fill}" stroke="{edge}"/>')

    def text(x, y, value, size=15, weight=400, color="#263644", anchor="start"):
        parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" style="fill:{color}">{escape(str(value))}</text>')

    def wire(path, arrow=True, dashed=False):
        tip = ' marker-end="url(#end)"' if arrow else ""
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        parts.append(f'<path class="wire" d="{path}"{tip}{dash}/>')

    def node(x, y, width, height, label, fill, sub="", size=15):
        box(x, y, width, height, fill)
        center = x + width / 2
        text(center, y + height / 2 + (-2 if sub else size * .34), label, size, 650, anchor="middle")
        if sub:
            text(center, y + height / 2 + 15, sub, 13, anchor="middle")

    def badge(x, y, number):
        box(x, y, 25, 24, ACCENTS[number - 1], edge="none")
        text(x + 12.5, y + 17, number, 14, 700, "white", "middle")

    def sheet(x, y, width, depth, label, face="#fff"):
        parts.append(f'<path d="M{x} {y}l{depth} {-depth}h{width}l{-depth} {depth}z" fill="{face}" stroke="#4388bc"/>')
        parts.append(f'<path d="M{x} {y}h{width}v15h{-width}z" fill="#fff" stroke="#4388bc"/>')
        parts.append(f'<path d="M{x + width} {y}l{depth} {-depth}v15l{-depth} {depth}z" fill="#c5d2de" stroke="#4388bc"/>')
        text(x + width / 2, y + 12, label, 12, 650, anchor="middle")

    box(0, 0, 1400, 1430, "#fff", 0, "none")
    text(47, 51, name, 38, 700, "#0088d1")
    text(360, 50, "论文模型 · 稀疏检测" if esod else "作者基线 · 全图检测", 22, 600, INK)
    text(1351, 49, "VisDrone  /  YOLOv5m  /  水平框", 17, 600, "#428abb", "end")
    box(38, 65, 1324, 1322, "#ededed", 17, "#d7dadd")
    for boundary in ("M446 66V1386", "M446 239H1361", "M446 827H1361", "M39 353H445"):
        wire(boundary, False)

    # 顶部以立体特征图概括整张网络；三条侧线从实际尺度流向融合节点。
    text(50, 94, "Backbone", 24, 700, INK)
    text(51, 116, "CSP · YOLOv5m", 15, 500, INK)
    text(51, 136, "三层侧输出", 15, 500, INK)
    text(462, 94, "Head", 24, 700, INK)
    text(538, 94, "FPN → PAN → Detect → 整图水平框", 16, 500, INK)
    sheet(69, 327, 262, 75, "无人机输入 · 街景示意", "url(#street)")
    sheet(96, 262, 220, 55, "P3 · stride 8")
    sheet(122, 206, 177, 43, "P4 · stride 16")
    sheet(151, 153, 133, 34, "P5 · stride 32")
    for start, turn, end, scale in [((292, 160), 500, 134, "P5"), ((307, 213), 520, 166, "P4"), ((324, 269), 430, 199, "P3")]:
        wire(f"M{start[0]} {start[1]}H{turn}V{end}H566")
        node(568, end - 10, 64, 20, scale, YELLOW, size=12)
        wire(f"M632 {end}H663")
        node(665, end - 10, 78, 20, "FPN / PAN", BLUE, size=12)
        wire(f"M743 {end}H790")
    for index, scale in enumerate(("P3", "P4", "P5")):
        sheet(791 + index * 6, 195 - index * 26, 74 - index * 5, 22, scale)
    wire("M879 166H927")
    node(930, 143, 105, 46, "SparseHead" if esod else "Detect", "url(#detect)", "三尺度预测", 15)
    wire("M1035 166H1056")
    node(1058, 143, 85, 46, "解码", "url(#detect)", "锚框 + 网格")
    box(1170, 94, 177, 128, "#cfdaeb", 9)
    wire("M1143 166h13v-42h23")
    wire("M1156 166v23h23")
    node(1181, 108, 74, 32, "BBox", "#fff", size=14)
    node(1181, 174, 74, 32, "Class", "#fff", size=14)
    text(1262, 129, "4 个坐标", 13)
    text(1262, 195, "10 类", 13)
    text(1258, 160, "目标性 × 类别分数", 13, 600, INK, "middle")

    # 左列标明真实通道数和重复数；P3 同时进入热图支路和切片支路。
    text(58, 381, "整图输入", 18, 700, INK)
    text(161, 381, "RGB · 长边 1536 · 等比例缩放", 14)
    wire("M201 342V360 M230 387V405")
    stages = [
        (407, "Focus · 3×3 Conv", "重排邻近像素后卷积", "P1", BLUE, "步长 2 × 48 通道"),
        (485, "Conv ↓2 → C3 ×2", "3×3 卷积下采样", "P2", BLUE, "步长 4 × 96 通道"),
        (563, "Conv ↓2 → C3 ×6", "整图浅层特征", "P3", YELLOW, "步长 8 × 192 通道"),
        (656, "DWConv → Segmenter" if esod else "P3 直通", "13×13 卷积预测热图" if esod else "保留完整特征图", "①", ORANGE, "单通道热图" if esod else "整幅 P3"),
        (754, "HeatMapParser" if esod else "整图传递", "从 P3 裁出候选特征块" if esod else "所有位置进入深层", "②", AQUA, "P3 块 × 192 通道" if esod else "整图 × 192 通道"),
        (850, "3×3 Conv ↓2", "空间降采样", "P4", BLUE, "步长 16 × 384 通道"),
        (953, "C3 ×6", "聚合中层上下文", "P4", YELLOW, "步长 16 × 384 通道"),
        (1054, "3×3 Conv ↓2", "空间降采样", "P5", BLUE, "步长 32 × 768 通道"),
        (1155, "SPP", "池化核 3 / 5 / 7" if esod else "池化核 5 / 9 / 13", "P5", GREEN, "拼接多尺度池化结果"),
        (1252, "C3 ×2", "输出深层特征", "P5", YELLOW, "步长 32 × 768 通道"),
    ]
    for index, (top, label, note, stage, fill, shape) in enumerate(stages):
        box(120, top, 223, 47, fill)
        wire(f"M301 {top}v47", False)
        text(130, top + 19, label, 15, 700)
        text(130, top + 38, note, 13)
        if index in (3, 4):
            badge(309, top + 11, index - 2)
        else:
            text(322, top + 28, stage, 13, 650, anchor="middle")
        text(239, top + 66, shape, 12)
        if index + 1 < len(stages):
            wire(f"M230 {top + 47}V{stages[index + 1][0] - 2}")
    if esod:
        wire("M120 588H88V778H118")
        text(60, 632, "P3", 13, 650, INK)
    text(58, 1344, "尺度标注使用原图坐标系；× 表示瓶颈重复数。", 13)
    text(230, 1375, "Backbone", 24, 700, INK, "middle")

    # 中部按参考图拆出 C3、通道表、三个带编号的机制及检测头支路。
    text(457, 267, "Details", 24, 700, INK)
    box(455, 278, 321, 537, YELLOW)
    text(466, 302, "C3 · CSP Bottleneck", 20, 700, INK)
    text(466, 324, "两路并行变换，在通道维度拼接。", 14)
    text(609, 364, "输入特征 x", 16, 650, anchor="middle")
    wire("M609 372v11H545v19")
    wire("M609 383h93v19")
    node(480, 405, 130, 43, "1×1 Conv", BLUE)
    node(641, 405, 122, 43, "1×1 Conv", BLUE)
    wire("M545 448v32")
    node(471, 482, 148, 55, "Bottleneck ×n", ORANGE, "1×1 → 3×3", 15)
    wire("M545 537v46h63v32")
    wire("M702 448v135h-78v32")
    node(554, 617, 126, 38, "Concat", GREEN)
    wire("M617 655v26")
    node(554, 683, 126, 38, "1×1 Conv", BLUE)
    wire("M617 721v20")
    text(617, 762, "输出特征 y", 16, 650, anchor="middle")
    text(466, 793, "YOLOv5m 的深度倍率 0.67，宽度倍率 0.75。", 13)
    columns = (787, 925, 1041, 1154, 1253, 1349)
    for row, entries in enumerate((("主干阶段", "P2", "P3", "P4", "P5"), ("C3 重复数", 2, 6, 6, 2),
                                   ("输出通道", 96, 192, 384, 768), ("输入步长", 4, 8, 16, 32))):
        for col, entry in enumerate(entries):
            left, right = columns[col:col + 2]
            top = 278 + row * 23
            box(left, top, right - left, 23, "#dfeaf4" if row == 0 else "#f3f3f3", 0, "#aebed0")
            text((left + right) / 2, top + 16, entry, 13, 650 if row == 0 else 400, anchor="middle")
    mechanisms = (
        (("ObjSeeker", "预测前景位置", "整图 P3", "DWConv 13×13", "Segmenter", "单通道响应", "热图监督", "由目标掩码学习"),
         ("AdaSlicer", "减少背景计算", "热图候选中心", "聚集邻近响应", "共享 P3 特征", "按区域切片", "特征块 + 偏移", "深层只处理所选块"),
         ("SparseHead", "裁片内稀疏预测", "热图局部峰值", "阈值与邻域扩展", "逐尺度索引", "选择有效位置", "解耦预测", "分类和框回归"))
        if esod else
        (("全图 P3", "保留浅层细节", "整图特征", "步长 8", "浅层侧输出", "参与后续融合", "全部位置", "进入深层主干"),
         ("稠密主干", "在全图上提取特征", "整幅 P3", "不筛前景位置", "卷积下采样", "P4 与 P5", "完整特征图", "背景同样计算"),
         ("稠密 Detect", "每个网格均预测", "三尺度特征", "逐位置卷积", "解耦预测", "分类和框回归", "解码候选框", "全图网格与锚框"))
    )
    for index, ((left, width), labels) in enumerate(zip(((787, 181), (976, 183), (1167, 182)), mechanisms)):
        title, sub, *steps = labels
        box(left, 381, width, 272, ("#f5cda9", "#9dc7ed", "#aaebf3")[index] if esod else "#dde3e7")
        text(left + 10, 407, title, 18, 700, INK)
        text(left + 10, 430, sub, 14)
        badge(left + width - 35, 389, index + 1)
        for step, fill in enumerate((BLUE, PURPLE, GREEN)):
            top = 440 + step * 65
            node(left + 11, top, width - 22, 46, steps[step * 2], fill, steps[step * 2 + 1], 14)
            if step < 2:
                wire(f"M{left + width / 2} {top + 46}v16")
        text(left + width / 2, 637, ("浅层定位", "深层计算范围", "检测头计算范围")[index], 13, 600, anchor="middle")
    box(787, 662, 562, 153, "#d5dfa1")
    text(798, 688, "Detect · 解耦头", 19, 700, INK)
    text(798, 709, "每尺度 3 个锚框，每个锚框输出 15 个数。", 14)
    node(798, 742, 100, 44, "输入特征", BLUE)
    wire("M898 764h28v-26h20")
    wire("M926 764v24h20")
    node(949, 722, 120, 32, "分类支路", AQUA)
    node(949, 772, 120, 32, "回归支路", ORANGE)
    wire("M1069 738h38")
    wire("M1069 788h38")
    node(1110, 722, 222, 32, "10 类分数", PURPLE)
    node(1110, 772, 222, 32, "4 个框参数 + 1 个目标性", YELLOW, size=14)

    # 下半部才是逐层计算图。左列三条侧线接在 YAML 的真实取值层；
    # 上采样走左列，下采样走右列，避免把三个检测尺度误画成串行。
    text(457, 854, "FPN / PAN", 21, 700, INK)
    text(596, 853, "自顶向下 ↑     自底向上 ↓", 14)
    text(974, 854, "Detect  /  整图输出", 20, 700, INK)
    rows = (915, 1066, 1217)
    for index, (level, channels, source_y, bend) in enumerate(((3, 192, 777, 378), (4, 384, 976, 392), (5, 768, 1275, 407))):
        center = rows[index]
        wire(f"M343 {source_y}H{bend}V{center}H472")
        node(474, center - 21, 93, 42, f"P{level}", BLUE, f"{channels} 通道")
        wire(f"M567 {center}H598")
        node(600, center - 21, 127, 42, "Concat → C3" if level < 5 else "1×1 Conv", YELLOW, ("P3 融合", "再经 1×1 降维", "P5 降维")[index], 14)
        wire(f"M727 {center}H781")
        node(783, center - 21, 146, 42, "P3 检测特征" if level == 3 else "Concat → C3", BLUE, f"{channels} 通道", 14)
        wire(f"M929 {center}H984")
        node(986, center - 21, 149, 42, f"P{level} · Detect", AQUA, "稀疏位置" if esod else "全图网格")
        if level != 5:
            mid = center + 75
            wire(f"M664 {center + 151 - 21}V{mid + 18}")
            node(605, mid - 16, 118, 32, "上采样 ×2", GREEN, size=14)
            wire(f"M664 {mid - 16}V{center + 23}")
            wire(f"M856 {center + 21}V{mid - 18}")
            node(797, mid - 16, 118, 32, "3×3 Conv ↓2", ORANGE, size=14)
            wire(f"M856 {mid + 16}V{center + 151 - 23}")
    badge(988, 944, 3)
    wire("M1135 915h19v150h24")
    wire("M1135 1066h43")
    wire("M1135 1217h19v-150")
    node(1181, 1041, 157, 50, "解码候选框", PURPLE, "合并三个尺度")
    wire("M1259 1091v34")
    node(1181, 1128, 157, 50, "还原整图坐标", BLUE, "叠加切片偏移" if esod else "缩放网格坐标")
    wire("M1259 1178v35")
    node(1181, 1216, 157, 43, "NMS", "url(#detect)", "抑制重复框")
    wire("M1259 1259v31")
    node(1166, 1293, 182, 43, "水平框 · 类别 · 分数", AQUA, size=14)
    text(471, 1306, "训练监督", 16, 650, INK)
    text(471, 1331, "框回归、目标性、分类" + ("；另加前景热图监督。" if esod else "。"), 14)
    text(884, 1375, "Head", 24, 700, INK, "middle")
    scores = "36.0 AP · 28.3 APsmall · 36.4 FPS" if esod else "36.2 AP · 28.5 APsmall · 26.1 FPS"
    text(40, 1411, f"作者报告 · VisDrone val · 长边 1536 · {scores}（表 III）。结构依据作者提交 bde3571。", 14, 400, "#536570")
    parts.append("</svg>")
    path = DOCS / "esod" / ("esod-architecture.svg" if esod else "yolov5m-baseline-architecture.svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    for use_esod in (False, True):
        draw(use_esod)
