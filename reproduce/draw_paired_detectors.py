"""使用统一的 S³OD 画布绘制本机配对实验，图中注明原论文使用的检测器。"""

from html import escape
from pathlib import Path
import sys


# 名称、论文检测器、本机骨干、输入、两组改动，以及用于解释改动的训练或推理细节。
PAPERS = {
    'scahi': ('SCAHI', 'YOLO11s（扩展实验）', 'YOLO11s', '整图 1280 + 重叠切片 1024', 'F0 · 框级融合', 'F2 · 稠密融合',
              '逐视图得到候选框，再按 IOS 阈值 0.8 去重。', '先按覆盖与边界给稠密证据加权，再在尺度空间取峰并输出框。',
              '先把每个视图变成单独的框，会损失跨视图的互补证据。', '相同整图、切片、权重与前向次数；用作者冻结配置，不在 val 调参。'),
    'daasd': ('DAASD', 'YOLO26n（表 4）', 'YOLO11s', '局部切片 → 640 × 640', '固定 640 / 重叠 20%', '按密度选切片',
              '每张图都用相同的切片尺寸和重叠比例。', '整图扫一次，用 4×3 网格计数；按密度选 480 / 640 / 800。',
              '固定切片可能在稀疏图上浪费前向，在拥挤图上又看不清。', '仅合并切片框，密度扫描框不输出；跨片类内 NMS IoU 0.5。'),
    'asahi': ('ASAHI', 'TPH-YOLOv5', 'YOLO11s', '整图与切片 → 512 × 512', '固定 512 像素切片', '按分辨率选网格',
              '固定边长，步长推进到边缘时可能多看一列或一行。', '长边不超过 1818 像素用 6 块，否则用 12 块；反推块的宽高。',
              '固定步长与图像尺寸不整除会增加冗余覆盖。', '重叠目标 15%，都保留整图，都用普通 NMS；仅比较网格策略。'),
    'butr': ('BUTR', 'BUTR-S / L · HGNetv2', 'HGNetv2', '整图 → 800 × 800', '固定解码查询预算', '动态解码查询预算',
             '每张图恒用 250 或 500 个查询进入同一个解码器。', '根据编码器类别置信度估计目标数，在 250～500 之间取整数。',
             '固定预算在稀疏图上可能浪费计算，在密集图上却不够。', '作者 train+val 权重，仅测 test-dev；同一尺度内配对，不对比 S 与 L 的绝对 AP。'),
    'asanms': ('ASA-NMS', '作者独立后处理函数', 'YOLO11s', '整图 → 640 × 640', 'IoU 硬 NMS', 'ASA 空间抑制',
               '按同类框的交并比抑制，阈值固定为 0.7。', '用两个角点的距离、最高分框的对角长度与置信度判定是否重复。',
               '密集布局中，同类邻居与重复框都可能互相重叠。', '两臂共享单视图和原始候选；分数阈值 0.001，每图最多 500 框。'),
    'softnms': ('Soft-NMS', 'R-FCN 等', 'YOLO11s', '整图 → 640 × 640', '硬 NMS', '高斯 Soft-NMS',
                '同类框 IoU 超过 0.7 时移除低分框。', '按 exp(−IoU² / 0.5) 降分，重新选最高分框。',
                '拥挤时，相邻车辆的两个真框也可能重叠。', '单次前向，同一批原始候选。'),
    'wbf': ('WBF', 'EfficientDet-B7', 'YOLO11s', '整图和水平镜像 → 640', '跨视图 NMS', '跨视图 WBF',
            '同类框按分数排序，冲突时保留最高分坐标。', '按置信度加权平均同类匹配框的四个角坐标。',
            '同一辆车在镜像与原图中框的位置可能有偏差。', '两臂输入相同，镜像还原后匹配 IoU 均为 0.55。'),
    'snip': ('SNIP', 'R-FCN', 'YOLO11s', '同一整图 → 三尺度金字塔', '合并全部尺度', '按目标尺寸筛框',
             '只去除无效框，全部视图一起参与跨尺度 NMS。', '大视图选小框，中视图选中框，小视图选大框；区间重叠。',
             '同一物体在不同缩放下，占用的特征像素差别很大。', '三次前向均保留；还原原图坐标后筛框，最后 NMS。'),
    'remdet': ('RemDet', 'YOLOv8-S（表 3）', 'YOLOv8s', '等比缩放 + 补边至 640', 'CSPDarknet', 'RemNet',
               '常规 YOLOv8s 骨干逐级下采样，C2f 提取特征。', 'RemNet 用 CED 下采样、GatedFFN 提取特征，再交给相同颈部。',
               '下采样后小目标信息容易丢失；作者改动了特征提取。', '同一组 64 张训练图，30 轮，仅切换作者骨干配置。'),
    'nwd': ('NWD', 'Faster R-CNN', 'YOLOv8s', '等比缩放 + 补边至 640', 'CIoU 框损失', 'NWD 框损失',
            '正样本回归按 CIoU 计算误差。', '将中心和半宽高的距离换算到输入像素，以 C=12.8 归一化。',
            '小框仅偏移几个像素，交并比就可能急剧下降。', '保持分类、分配和 DFL 不变，仅用正样本框回归项训练。'),
    'focal': ('Focal Loss', 'RetinaNet', 'ResNet-50', '整图等比缩入 640 × 640', 'α 平衡交叉熵', 'Focal Loss',
              'γ=0，α=0.75，采用论文给出的较强交叉熵配置。', 'γ=2，α=0.25，分类项乘以 (1−pₜ)²。',
              '密集检测中，容易判别的背景锚框数量远多于前景。', '匹配规则、L1 回归和推理流程在两组间一致。'),
}


def render(paper, method):
    title, original, backbone, view, base, change, base_detail, detail, problem, protocol = PAPERS[paper]
    choice = change if method else base
    stage = 1 if paper in ('asahi', 'daasd') else 3 if paper == 'butr' else 2 if paper == 'remdet' else 4 if paper in ('focal', 'nwd') else 5
    backbone_label = choice if paper == 'remdet' else backbone
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1430" viewBox="0 0 1400 1430" role="img">',
                f'<title>{title} · {choice} · VisDrone 本机配对</title>',
                '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10" fill="#527480"/></marker></defs>',
                '<rect width="1400" height="1430" fill="white"/><g font-family="Microsoft YaHei, sans-serif">']

    def text(left, top, label, size=22):
        elements.append(f'<text x="{left}" y="{top}" font-size="{size}" fill="#114d6b">{escape(label)}</text>')

    def box(left, top, width, height, heading, body, color):
        elements.append(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" rx="7" fill="{color}" stroke="#8ea2aa"/>')
        text(left + 14, top + 30, heading)
        text(left + 14, top + 59, body, 18)

    def link(path):
        elements.append(f'<path d="{path}" fill="none" stroke="#527480" stroke-width="2" marker-end="url(#arrow)"/>')

    text(47, 51, title, 38)
    text(360, 50, f'本机配对 · {choice}', 26)
    text(1050, 50, 'VisDrone  /  十类水平框')
    elements.append('<rect x="38" y="65" width="1324" height="1322" rx="17" fill="#ededed" stroke="#c8ced1"/><path d="M446 65V1387 M446 239H1362 M446 827H1362 M38 353H446" fill="none" stroke="#a8b4ba"/>')
    text(52, 104, f'Backbone · {backbone_label}')
    text(462, 104, 'Neck / Head · 多尺度预测')
    for index, level in enumerate(('浅层', '中层', '深层')):
        left, top = 80+index*32, 297-index*54
        elements.append(f'<path d="M{left} {top}l18 -19h165l-18 19z" fill="white" stroke="#478db9"/>')
        text(left+36, top-4, level+'特征', 15)
        link(f'M{left+184} {top-12}H{365+index*21}V{165+index*5}H483')
    stages = [('特征融合', '融合不同分辨率'), ('检测头', '分类与边框回归'), ('解码', '候选框与分数')]
    if paper == 'scahi':
        stages[-1] = ('稠密头输出', '类别概率与边框分布')
    elif paper == 'butr':
        stages = [('多尺度编码', '4 层特征'), ('预算 → top-k', '选择解码查询'), ('查询解码', '坐标与类别分数')]
    for left, (heading, body) in zip((485, 790, 1095), stages):
        box(left, 130, 230, 78, heading, body, '#c2e4be')
        if left < 1095:
            link(f'M{left+230} 169h70')
    text(54, 394, '整图检测路径')
    steps = [('输入', 'VisDrone 航拍图'), ('构造视图', view), ('骨干网络', backbone_label),
             ('融合与预测', '多层特征 → 类别和坐标'), ('训练监督', '水平框类别与位置' if paper in ('focal','nwd','remdet') else '固定检测器权重'),
             ('输出候选', '映射回原图，按类别处理'), ('输出水平框', '坐标、类别、置信度')]
    if paper in ('scahi', 'daasd', 'asahi', 'butr', 'asanms'):
        steps[4] = ('解码查询', '输出类别和位置') if paper == 'butr' else ('检测头输出', '稠密类别与边框分布' if paper == 'scahi' else '各位置的类别与坐标')
        text(468, 790, '训练监督：已有水平框检测器权重；本实验仅推理，不反传。', 19)
    for index, (heading, body) in enumerate(steps):
        top = 423+index*130
        box(75, top, 326, 80, '① '+heading if index == stage else heading, choice if index == stage else body,
            '#ffe5a0' if index == stage else '#d4e4f1')
        if index < 6:
            link(f'M238 {top+80}v47')
    link(f'M401 {463+stage*130}H425V445H478')
    text(469, 289, 'Details · 对照变量')
    box(481, 337, 839, 82, '原论文观察的问题', problem, '#d5e7f1')
    box(481, 445, 839, 82, '① '+choice, detail if method else base_detail, '#ffe5a0')
    box(481, 559, 839, 82, '固定条件', protocol, '#bae3dc')
    box(481, 674, 839, 82, '原论文检测器', original, '#d8d1ea')
    text(469, 880, '输出与评测')
    link('M401 1243H448V992H478')
    output = '从融合后的峰值读取目标框，按分数最多保留 500 框。' if paper == 'scahi' and method else '按查询解码并筛出类别 1–10，无切片、无 NMS。' if paper == 'butr' else '经过类内去重并限制每张图的预测数量。'
    box(481, 954, 839, 82, '十个类别的整图水平框', output, '#c2e4be')
    link('M898 1036v68')
    box(481, 1108, 839, 82, '同机配对评测', '按原图面积计算 AP_small，在相同图像上记录整图毫秒数。', '#ffe5a0')
    return '\n'.join([*elements, '</g></svg>', ''])


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1] / 'docs/reproduce'
    for paper in (sys.argv[1:] or PAPERS):
        folder = root / paper
        folder.mkdir(exist_ok=True)
        for method in (False, True):
            name = 'method' if method else 'baseline'
            (folder / f'{name}-architecture.svg').write_text(render(paper, method), encoding='utf-8')
