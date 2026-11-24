from html import escape
from pathlib import Path


destination = Path("docs/paper/figures/s3od-architecture.svg")
svg = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1430" viewBox="0 0 1400 1430" role="img" aria-labelledby="title desc">
<title id="title">S³OD 检测器逐层结构与半监督训练模块</title>
<desc id="desc">左侧为 ResNet-50 的全部四个阶段，右上为特征与损失总览，中部展开瓶颈、SAT、SLA、TNL 和 RoI 预测头，下部按尺度连接 FPN、RPN、RoIAlign 与旋转框输出。图中空间尺寸以 1024 方形输入举例。</desc>
<defs>
  <marker id="end" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5.6" markerHeight="5.6" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#576775"/></marker>
  <linearGradient id="detect" x2="1" y2="1" gradientUnits="objectBoundingBox"><stop stop-color="#aaca8e"/><stop offset="1" stop-color="#ffdc65"/></linearGradient>
  <pattern id="airfield" patternUnits="userSpaceOnUse" width="184" height="90"><rect width="184" height="90" fill="#adb9a1"/><path d="M-4 5L186 61M-5 42L186 98" stroke="#788980" stroke-width="12"/><path d="M-4 5L186 61" stroke="#e6e4cf" stroke-width="1.3" stroke-dasharray="10 5"/><path d="M8 78l34-48m54 35 20-27" stroke="#d3cbb2" stroke-width="8"/><g fill="#eff0df" stroke="#46555a" stroke-width=".65"><path id="plane" d="M0-8l1.5 6 7 4v2L1 2v4l2.5 2v1L0 8l-3.5 1V8l2.5-2V2l-7.5 2V2l7-4z"/><use href="#plane" transform="translate(49 59) rotate(42)"/><use href="#plane" transform="translate(85 68) rotate(42)"/><use href="#plane" transform="translate(135 24) rotate(-32)"/></g></pattern>
</defs>
<style>text{font-family:"Microsoft YaHei",sans-serif;fill:#263644} .wire{fill:none;stroke:#71808a;stroke-width:1.35;stroke-linecap:round;stroke-linejoin:round}</style>''']
ink = "#09557e"
blue, yellow, orange, lime = "#85d9f1", "#ffe8a2", "#f6cba6", "#bee09c"
purple, aqua, gold, muted = "#e6dff5", "#b9eeea", "#ffcf3e", "#536570"
stroke = "#8c9da6"


def rect(x, y, w, h, fill, radius=4, edge=stroke, sw=1):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{edge}" stroke-width="{sw}"/>')


def txt(x, y, value, size=15, weight=400, color="#263644", anchor="start"):
    svg.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" style="fill:{color}">{escape(str(value))}</text>')


def path(d, arrow=True, dashed=False):
    marker = ' marker-end="url(#end)"' if arrow else ''
    dash = ' stroke-dasharray="5 4"' if dashed else ''
    svg.append(f'<path class="wire" d="{d}"{marker}{dash}/>')


def module(x, y, w, h, title, fill, note="", size=16):
    rect(x, y, w, h, fill, 4)
    txt(x+w/2, y+h/2+(0 if note else size*.34)-2*bool(note), title, size, 650, anchor="middle")
    if note:
        txt(x+w/2, y+h/2+15, note, 13, anchor="middle")


def feature(x, y, width, depth, label, image=False):
    surface = "url(#airfield)" if image else "#fff"
    svg.append(f'<path d="M{x} {y}l{depth} {-depth}h{width}l{-depth} {depth}z" fill="{surface}" stroke="#4388bc"/>')
    svg.append(f'<path d="M{x} {y}h{width}v15h{-width}z" fill="#fff" stroke="#4388bc"/><path d="M{x+width} {y}l{depth} {-depth}v15l{-depth} {depth}z" fill="#c5d2de" stroke="#4388bc"/>')
    txt(x+width/2, y+11.8, label, 11.5, 650, anchor="middle")


def panel(x, y, w, h, fill, title, subtitle):
    rect(x, y, w, h, fill)
    txt(x+10, y+23, title, 19, 700)
    txt(x+10, y+44, subtitle, 13.5)


rect(0, 0, 1400, 1430, "#fff", 0, "none")
txt(47, 51, "S³OD", 38, 700, "#0088d1")
txt(171, 50, "航拍旋转目标检测", 22, 600, ink)
txt(1351, 49, "DOTA-v1.5   /   Rotated Faster R-CNN", 17, 600, "#428abb", "end")
rect(38, 65, 1324, 1322, "#ededed", 17, "#d7dadd", 2.8)
for divider in ["M446 66V1386", "M446 239H1361", "M446 827H1361", "M39 353H445"]:
    path(divider, False)
txt(50, 94, "Backbone", 24, 700, ink)
txt(51, 115, "ResNet-50", 15, 500, ink)
txt(51, 135, "四级特征", 15, 500, ink)
txt(462, 94, "Head", 24, 700, ink)
txt(538, 94, "FPN → RPN → RoI Head", 16, 500, ink)

# Mirrored overview of the same feature paths expanded in the lower half.
feature(69, 327, 262, 75, "航拍输入示意", True)
for xx, yy, ww, dd, name in [(92,273,229,56,"C2"), (111,230,196,48,"C3"), (135,190,157,39,"C4"), (155,151,120,30,"C5")]:
    feature(xx, yy, ww, dd, name)
for idx, (yy, endx, endy) in enumerate([(159,540,160),(198,562,177),(238,585,194),(281,609,211)]):
    path(f"M{[305,331,355,377][idx]} {yy}H{endx-34}V{endy}H{endx}")
    module(endx, endy-10, 58, 20, "1×1", yellow, size=12)
    path(f"M{endx+58} {endy}H690")
    module(690, endy-10, 61, 20, "FPN", blue, size=12)
    path(f"M751 {endy}H816")
for idx in range(5):
    feature(799+idx*4, 204-idx*16.5, 93-idx*7, 23, "P"+str(idx+2))
path("M911 170H937")
module(940, 146, 81, 45, "RPN", "url(#detect)", "候选框")
path("M1021 168H1037")
module(1040, 146, 96, 45, "RoI Head", "url(#detect)", "分类与定位")
path("M1136 168h13v-40h21")
path("M1149 168v26h21")
rect(1170, 91, 174, 132, "#cfdaeb", 10)
module(1182, 107, 80, 38, "BBox", "#fff", size=14)
module(1182, 176, 80, 35, "Class", "#fff", size=14)
txt(1268, 129, "Smooth L1", 12, 600)
txt(1270, 198, "CE", 13, 600)
txt(1256, 162, "RoI Loss", 13, 650, ink, "middle")

# Full-height backbone, with stage identifiers and shapes on the actual edges.
txt(58, 381, "输入裁片", 18, 700, ink)
txt(168, 380, "1024 × 1024 × 3（尺寸示例）", 14)
left_stages = [
    (407, "Conv1", "7×7，s=2；BN + ReLU", "Stem", blue, "512 × 512 × 64"),
    (485, "MaxPool", "3×3，s=2；下采样", "Stem", "#c1d6f6", "256 × 256 × 64"),
    (563, "Bottleneck ×1", "投影捷径，扩张通道", "C2", yellow, "256 × 256 × 256"),
    (656, "Bottleneck ×2", "恒等捷径，保留细节", "C2", yellow, "256 × 256 × 256"),
    (754, "Bottleneck ×1", "投影捷径，s=2", "C3", yellow, "128 × 128 × 512"),
    (850, "Bottleneck ×3", "恒等捷径，细化特征", "C3", yellow, "128 × 128 × 512"),
    (953, "Bottleneck ×1", "投影捷径，s=2", "C4", yellow, "64 × 64 × 1024"),
    (1054, "Bottleneck ×5", "恒等捷径，累积上下文", "C4", yellow, "64 × 64 × 1024"),
    (1155, "Bottleneck ×1", "投影捷径，s=2", "C5", yellow, "32 × 32 × 2048"),
    (1252, "Bottleneck ×2", "恒等捷径，输出深层语义", "C5", yellow, "32 × 32 × 2048"),
]
path("M231 354V361 M231 387V405")
for idx, (yy, name, desc, stage, color, shape) in enumerate(left_stages):
    rect(120, yy, 223, 47, color, 5)
    path(f"M301 {yy}v47", False)
    txt(130, yy+19, name, 16, 700)
    txt(130, yy+38, desc, 13)
    txt(322, yy+28, stage, 13, 650, anchor="middle")
    if idx < len(left_stages)-1:
        path(f"M230 {yy+47}V{left_stages[idx+1][0]-2}")
    txt(240, yy+66, shape, 12.5)

# Details panel: one tall block, a table above three narrow mechanism diagrams,
# and a long detection-head inset below, as in the supplied reference.
txt(457, 267, "Details", 24, 700, ink)
panel(455, 278, 321, 537, "#ffe8a2", "Bottleneck", "三层卷积与一条捷径构成残差单元")
txt(570, 359, "输入特征  x", 15, 650, anchor="middle")
path("M570 367v24")
for yy, name, sub in [(392,"1×1 Conv","压缩通道；首块可下采样"),(470,"3×3 Conv","提取局部空间特征"),(548,"1×1 Conv","恢复到 4 倍中间通道")]:
    module(482, yy, 177, 51, name, blue, sub)
    if yy < 548:
        path(f"M570 {yy+51}v25")
path("M570 373H712V480")
module(670, 482, 91, 67, "Shortcut", orange, "恒等 / 投影", 14)
path("M712 549V631H590")
path("M570 599v15")
svg.append('<circle cx="570" cy="631" r="15" fill="#fff8e2" stroke="#ae9e62"/>')
txt(570, 637, "+", 20, 600, anchor="middle")
path("M570 646v22")
module(507, 670, 126, 36, "ReLU", aqua)
path("M570 706v19")
txt(570, 748, "输出特征  y", 16, 650, anchor="middle")
txt(467, 779, "同尺寸时直接相加；尺寸或通道改变时，", 13)
txt(467, 799, "捷径用 1×1 卷积对齐。", 13)
cols = [787, 925, 1041, 1154, 1253, 1349]
rows = [["主干阶段","C2","C3","C4","C5"],["瓶颈重复数","3","4","6","3"],["输出通道","256","512","1024","2048"],["输入步长","4","8","16","32"]]
for row_index, row in enumerate(rows):
    for col_index, entry in enumerate(row):
        xx, yy = cols[col_index], 278+row_index*23
        ww = cols[col_index+1]-xx
        rect(xx,yy,ww,23, "#dfeaf4" if row_index==0 else "#f3f3f3",0,"#aebed0",.8)
        txt(xx+ww/2, yy+16, entry, 13, 650 if row_index==0 else 400, anchor="middle")

def training_card(x, width, color, title, subtitle, blocks, foot):
    panel(x,381,width,272,color,title,subtitle)
    center = x+width/2
    for idx, (label, sub, fill) in enumerate(blocks):
        yy=440+idx*65
        module(x+11,yy,width-22,46,label,fill,sub,14)
        if idx<2:
            path(f"M{center} {yy+46}v16")
    txt(center,636,foot,12.5,600,anchor="middle")

training_card(787,181,"#f5cda9","SAT","按目标大小筛伪框",
              [("教师框 + 分数","框面积划分尺度组",blue),("组内估计阈值","10% 配置使用 GMM",purple),("保留高分伪框","坐标转到强增强视图",lime)],"接入点：教师输出")
training_card(976,183,"#9dc7ed","SLA","给小目标分配正锚框",
              [("框 → 二维高斯","编码中心、形状和朝向",aqua),("GWD 匹配","每个目标选 Top-2",yellow),("按尺度重加权","平衡两组正样本损失",lime)],"接入点：锚框分配 / 损失")
training_card(1167,182,"#aaebf3","TNL","复核学生负候选",
              [("教师复评候选","读取背景置信度",blue),("筛可信背景","教师背景分数 > 0.7",lime),("补充困难负样本","按置信度加权",orange)],"接入点：RoI 分类损失")
panel(787,662,562,153,"#d5dfa1","RoI Head","每个候选区域单独分类和回归")
module(798,729,99,55,"RoIAlign",blue,"7×7×256",14)
path("M897 756h15")
module(915,729,105,55,"FC ×2",blue,"各 1024 维",15)
path("M1020 756h13v-29h16")
path("M1033 756v27h16")
module(1051,710,130,33,"Cls · 17 个分数",aqua,size=13)
module(1051,767,130,33,"Reg · 16×5",orange,size=13)
path("M1181 727h16")
path("M1181 784h16")
module(1200,710,137,33,"类别 + 背景",purple,size=13)
module(1200,767,137,33,"各类旋转框偏移",purple,size=13)

# Layer graph: each backbone stage has its own lateral input. The FPN merges
# lateral features before the output convolutions; P6 is pooled from P5.
txt(457, 854, "FPN", 20, 700, ink)
txt(512, 853, "横向 1×1 → 自顶向下相加 → 3×3 平滑", 14)
txt(928, 854, "RPN  /  RoI Head", 19, 700, ink)
levels=[(2,890,679,368),(3,1000,873,381),(4,1110,1077,394),(5,1220,1275,408)]
for level, yy, stage_y, bend in levels:
    path(f"M343 {stage_y}H{bend}V{yy}H465")
    module(467,yy-17,107,34,"1×1 Conv",blue,size=15)
    path(f"M574 {yy}h29")
    module(606,yy-17,88,34,"横向输入" if level==5 else "Add",yellow if level==5 else gold,size=14)
    path(f"M694 {yy}h36")
    module(733,yy-20,120,40,"3×3 Conv",blue,f"P{level} · {1024//(2**level)}² × 256",14)
    if level>2:
        path(f"M705 {yy}v-26h-56v-8")
        module(594,yy-70,111,34,"Upsample ×2",lime,size=13)
        path(f"M649 {yy-70}v-21")
    path(f"M853 {yy}h27",False)
    svg.append(f'<circle cx="880" cy="{yy}" r="2.7" fill="#6a7c88"/>')
path("M793 1240v40")
module(733,1282,120,40,"MaxPool",purple,"P6 · 16² × 256",14)
path("M853 1302h27V890", False)
path("M880 891h48")
module(931,870,158,46,"共享 RPN","url(#detect)","P2–P6 生成候选")
path("M1089 893h43")
module(1135,870,198,46,"NMS","#e6def2","去除重复的水平候选框")
path("M1234 916v45")
module(1135,964,198,50,"RoIAlign",blue,"按区域从 P2–P5 取特征")
path("M854 1004h52v-16h226",True,True)
txt(930,981,"P2–P5 特征",12)
path("M1234 1014v37")
module(1135,1054,198,48,"Shared 2FC",yellow,"汇总候选区域的特征")
path("M1234 1102v26h-98v25")
path("M1234 1128h55v25")
module(1066,1155,137,49,"分类",aqua,"16 类 + 背景")
module(1212,1155,137,49,"回归",orange,"中心、宽高、角度")
path("M1135 1204v24h68v15")
path("M1281 1204v24h-78",False)
module(1092,1245,221,48,"旋转 NMS","url(#detect)","输出类别、分数、旋转框")
txt(916,1051,"训练时的监督来源",15,700,ink)
txt(916,1077,"有标签：人工旋转框",13)
txt(916,1098,"无标签：弱增强教师",13)
txt(916,1119,"指导强增强学生",13)
txt(916,1203,"更新学生参数",14,650,ink)
path("M970 1212v21")
module(914,1235,115,35,"EMA",purple,size=14)
path("M970 1270v19")
txt(970,1307,"更新教师参数",13,600,anchor="middle")
txt(471,1306,"P6 由 P5 下采样",13)
txt(471,1329,"P2–P6 输出均为 256 通道",13)
txt(56,1348,"尺寸标注：高 × 宽 × 通道；s 为步长。",13,400,muted)
txt(243,1373,"Backbone",24,700,ink,"middle")
txt(904,1373,"Head",24,700,ink,"middle")
txt(41,1410,"依据：S³OD 预印本 §III 与作者 DOTA-v1.5 配置 d7806b7。训练使用多尺度输入；图中方形裁片用于读出各层尺寸。",13,400,muted)
svg.append("</svg>")
destination.write_text("\n".join(svg)+"\n", encoding="utf-8")
