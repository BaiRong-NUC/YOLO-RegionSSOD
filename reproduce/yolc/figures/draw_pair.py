"""绘制 CenterNet 基线与 YOLC 的完整检测路径，仅输出 SVG。"""

from html import escape
from pathlib import Path


def render(local_scale):
    title = 'YOLC' if local_scale else 'CenterNet'
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1400 1430" width="1400" height="1430">',
             '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10" fill="#547685"/></marker></defs>',
             '<rect width="1400" height="1430" fill="white"/><g font-family="Microsoft YaHei, sans-serif">']

    def text(left, top, label, size=18):
        parts.append(f'<text x="{left}" y="{top}" font-size="{size}" fill="#09557e">{escape(label)}</text>')

    def box(left, top, width, height, heading, detail, color):
        parts.append(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" rx="6" fill="{color}" stroke="#9cabb2"/>')
        text(left + 15, top + 29, heading, 20)
        text(left + 15, top + 55, detail, 16)

    def arrow(path):
        parts.append(f'<path d="{path}" fill="none" stroke="#547685" stroke-width="2" marker-end="url(#arrow)"/>')

    # 复用 S³OD 的画布与分区坐标，主链从左侧输入连到右下的整图输出。
    text(47, 51, title, 38)
    text(360, 50, '论文方法 · 密集区局部放大' if local_scale else '论文基线 · 整图中心点检测', 24)
    text(1050, 50, 'VisDrone  /  十类水平框', 20)
    parts.append('<rect x="38" y="65" width="1324" height="1322" rx="17" fill="#ededed" stroke="#c8ced1"/><path d="M446 65V1387 M446 239H1362 M446 827H1362 M38 353H446" fill="none" stroke="#a8b4ba"/>')
    text(50, 98, 'Backbone · HRNet', 26)
    text(462, 98, 'Head · 预测水平框', 26)
    for index, label in enumerate(('高分辨率分支', '中分辨率分支', '低分辨率分支', '最深层分支')):
        top = 300 - 53 * index
        left = 72 + 18 * index
        parts.append(f'<path d="M{left} {top}l20 -18h195l-20 18z" fill="white" stroke="#478db9"/>')
        text(left + 42, top - 3, label, 14)
        arrow(f'M{left + 207} {top - 10}H{368 + 15 * index}V{160 + index * 6}H480')
    box(482, 120, 235, 80, '多尺度特征', '融合到高分辨率特征图', '#bee09c')
    arrow('M717 160H756')
    box(758, 120, 265, 80, '① 检测头' if local_scale else '检测头', '解耦热图、两步回归' if local_scale else '中心点、宽高与偏移', '#85d9f1')
    arrow('M1023 160H1064')
    box(1066, 120, 263, 80, '单个视图的检测', '中心与边界 → 类别、框', '#cfdaeb')
    text(54, 387, '从输入到检测结果', 23)
    steps = [('原始航拍图', '以完整图像为单位'), ('整图输入', '保留全图上下文'), ('HRNet 特征提取', '并行分支交换空间信息'),
             ('① 估计中心与边界' if local_scale else '估计中心与边界', '输出中心热图与候选框'),
             ('② 选择密集区' if local_scale else '解码整图预测', '16×10 网格中找相邻密集格' if local_scale else '从热图峰值获得中心点'),
             ('② 放大并再次检测' if local_scale else '水平框还原', '使用同一模型检测局部' if local_scale else '将边界映射回原图'),
             ('③ 合并两种视图' if local_scale else '汇集整图预测', '局部结果替换区域内整框' if local_scale else '每张图只检测一次')]
    for index, (heading, detail) in enumerate(steps):
        top = 405 + 134 * index
        box(85, top, 311, 74, heading, detail, '#f6cba6' if local_scale and index >= 4 else '#cfdaeb')
        if index < 6:
            arrow(f'M241 {top + 74}v58')
    text(461, 274, 'Details · 为什么这样做', 25)
    box(475, 315, 834, 81, '① 预测与监督' if local_scale else '整图检测的困难',
        '中心热图用焦点损失监督，粗框和可变形卷积精修框用 GWD 监督。' if local_scale else '拥挤处的小车只占少量像素，整图只给模型一次区分相邻目标的机会。', '#ffe8a2')
    arrow('M894 396v65')
    box(475, 463, 834, 81, '② 放大哪些地方？' if local_scale else '训练监督',
        '热图密度最高的 15 格合并成连通区域，最多选两个；过滤面积小于 96² 的区域。' if local_scale else '根据水平框生成中心点热图，用真值框约束宽高与中心偏移。', '#f6cba6')
    arrow('M894 544v65')
    box(475, 611, 834, 81, '③ 局部检测回到整图' if local_scale else '论文对照范围',
        '区域外扩 1.2 倍、裁片放大 1.5 倍。还原局部坐标后保留区域外粗框。' if local_scale else '表 I 的原始 CenterNet 是整模型基线；本机配对使用同一 YOLC 权重。', '#b9eeea')
    text(474, 762, '本机实验只切换局部重检，两组共享网络结构和权重。', 19)
    text(461, 869, '输出与评测', 25)
    arrow('M396 1246H422V960H489')
    box(492, 919, 813, 82, '整图水平框输出', '每个框记录位置、类别与置信度', '#bee09c')
    arrow('M899 1001v82')
    box(492, 1086, 813, 82, '同机比较精度与耗时', '在同一批图上比较小目标 AP、召回率与平均毫秒数', '#ffe8a2')
    text(492, 1273, '论文依据：表 I，VisDrone val；作者代码：configs/yolc.py、eval_yolc.py。', 17)
    return '\n'.join([*parts, '</g></svg>', ''])


if __name__ == '__main__':
    docs = Path(__file__).resolve().parents[3] / 'docs/reproduce/yolc'
    docs.mkdir(parents=True, exist_ok=True)
    for name, local_scale in [('centernet-baseline', False), ('yolc', True)]:
        (docs / f'{name}-architecture.svg').write_text(render(local_scale), encoding='utf-8')
