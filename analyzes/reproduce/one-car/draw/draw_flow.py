"""生成单车教学图，并从种子 0 的实际检查点读取逐层形状。"""

import base64
from html import escape
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
CASE = '0000001_02999_d_0000005'
WEIGHT = ROOT / 'cache/2026-11-24-RegionSSOD/worktree/runs/partial10_b2_formal100_s0/weights/best.pt'
BLUE, GREEN, AMBER = '#e4eff9', '#e1f2ec', '#fff0da'


class Diagram:
    def __init__(self, width, height, title):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img"><title>{title}</title>',
                      '<defs><marker id="end" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 1L9 5L0 9Z" fill="#587181"/></marker></defs>',
                      '<g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif">']
        self.rect(0, 0, width, height, '#f7f9fa')

    def rect(self, left, top, width, height, color, radius=0):
        self.parts.append(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" rx="{radius}" fill="{color}"/>')

    def text(self, left, baseline, value, size=18, bold=False, color='#193849'):
        self.parts.append(f'<text x="{left}" y="{baseline}" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escape(value)}</text>')

    def card(self, left, top, width, height, color, heading, lines):
        self.rect(left, top, width, height, color, 10)
        self.text(left + 20, top + 34, heading, 23, True)
        for index, line in enumerate(lines):
            self.text(left + 20, top + 64 + index * 27, line)

    def wire(self, path):
        self.parts.append(f'<path d="{path}" fill="none" stroke="#587181" stroke-width="2" marker-end="url(#end)"/>')

    def save(self, name):
        folder = HERE / 'output'
        folder.mkdir(exist_ok=True)
        (folder / name).write_text('\n'.join(self.parts + ['</g></svg>']), encoding='utf-8')


def overview():
    diagram = Diagram(1190, 1396, 'YOLO11s 从训练图片到单车漏检的全过程')
    diagram.text(34, 52, '一辆车如何被学会，又如何被漏掉？', 34, True)
    diagram.text(35, 87, '左边读历史训练，右边读本次识别。两条路经过同一个网络结构。', 20)
    diagram.card(32, 115, 548, 192, BLUE, '训练输入｜带答案的裁片', [
        '6471 张人工标注裁片 + 4624 张离线伪标签裁片。',
        '每个目标给出类别、中心点、宽和高；坐标按图归一化。',
        '拼图、缩放、翻转时同步变换框；像素除以 255。',
        '一批通常 8 张 → [8, 3, 640, 640]，标签另送损失函数。'])
    diagram.card(610, 115, 548, 192, GREEN, '识别输入｜验证集上的这一张照片', [])
    # 红框标出本次选择的小车；缩略图来自原始照片，不作为模型输入。
    image = cv2.imread(str(ROOT / f'data/visdrone/images/val/{CASE}.jpg'))
    thumb = cv2.resize(image, (208, 117), interpolation=cv2.INTER_AREA)
    cv2.rectangle(thumb, (84, 2), (92, 11), (30, 30, 255), 1)
    encoded = base64.b64encode(cv2.imencode('.jpg', thumb)[1]).decode()
    diagram.parts.append(f'<image x="630" y="170" width="208" height="117" href="data:image/jpeg;base64,{encoded}"/>')
    for index, line in enumerate(['原图 1920 × 1080，红框处有车。', '整图缩至 640 × 360，上下各填 140。', 'RGB 像素除以 255，输入为', '[1, 3, 640, 640]，车宽约 8.33 像素。']):
        diagram.text(851, 184 + index * 29, line, 16)
    diagram.wire('M307 307V332'); diagram.wire('M884 307V332')
    diagram.card(32, 338, 1126, 155, '#e5eaef', '共同经过 YOLO11s 的第 0～23 层　↗ 逐层输入、输出与作用见配套图', [
        '0～10 提取特征 → 11～22 融合不同尺度 → 23 同时预测框的位置和类别。训练更新参数，识别加载 best.pt。',
        '第 16、19、22 层送出 80×80、40×40、20×20 三张特征图，共 6400 + 1600 + 400 = 8400 个候选位置。',
        '每个位置都输出 4 边 × 16 档距离和 10 个类别的原始分数。训练和识别随后走向不同的输出。'])
    diagram.wire('M307 493V518'); diagram.wire('M884 493V518')
    diagram.card(32, 524, 548, 160, BLUE, '训练 ①　让候选与标签配对', [
        '输入：预测分类、解码框、格点、人工框或伪标签。',
        '按格点位置、类别分数和重叠度挑正样本。',
        '输出：前景位置、各自负责的目标框和分类目标分数。'])
    diagram.card(610, 524, 548, 160, GREEN, '识别 ①　把网络数字变成候选框', [
        '四边距离分布 → 到格点的距离 → 框中心、宽、高。',
        '10 个分类值各自经过 sigmoid，变成 0～1 的分数。',
        '输出 [1, 14, 8400]：每个候选有 4 个框值 + 10 个分数。'])
    diagram.wire('M307 684V709'); diagram.wire('M884 684V709')
    diagram.card(32, 715, 548, 160, BLUE, '训练 ②　衡量这次猜得怎样', [
        '分类 BCE：每个位置的各类得分与目标分数差多少？',
        '前景框 CIoU：预测位置、大小与目标框差多少？',
        '前景 DFL：四条边的距离分布偏了多少？合成训练损失。'])
    diagram.card(610, 715, 548, 160, GREEN, '识别 ②　把候选变成最终输出', [
        '保留最高类分数 > 0.001 的候选，取其最高分类。',
        '同类框按分数去重（NMS IoU 0.7），全图取前 500。',
        '映回原图 → 每项是 [左, 上, 右, 下, 分数, 类别]。'])
    diagram.wire('M307 875V900'); diagram.wire('M884 875V900')
    diagram.card(32, 906, 548, 154, BLUE, '训练 ③　改权重，再看下一批', [
        '反向传播算梯度 → SGD 更新权重 → 下一批重新前向。',
        '从 COCO 预训练权重起步，历史运行共 100 轮。',
        '按验证表现保留 best.pt，供右侧识别时加载。'])
    diagram.card(610, 906, 548, 154, AMBER, '核对输出　车在整图里被找到了吗？', [
        '人工框 [800, 49, 825, 74]。认作 car 且 IoU ≥ 0.50 算找到。',
        '整图最终 car 框的最高 IoU 为 0.4400 → 漏检。',
        '沿右侧流程回看 8400 个候选，找到首次失去匹配的位置。'])
    diagram.wire('M884 1060V1083H307V1100'); diagram.wire('M884 1060V1100')
    diagram.card(32, 1105, 548, 197, AMBER, '回看整图｜有一个位置合格，却选成了人', [
        '仅 1 个候选的 IoU ≥ 0.50，来自第 16 层的 P3 特征。',
        '它的 IoU 是 0.5186；people 0.0596，car 0.0171。',
        '两类均过门槛，取最高分类时选了 people。',
        '走到同类去重之前，位置和类别都对的车框已有 0 个。'])
    diagram.card(610, 1105, 548, 197, GREEN, '对照切片｜同一权重找回这辆车', [
        '从原图取 640 方块，重叠 25%，并保留整图视图。',
        '各视图走同一网络，把框放回原图，再按类去重（0.5）。',
        '最终 car，分数 0.7391，IoU 0.8057。',
        '框约为 [801.0, 49.7, 826.2, 76.7]，匹配同一辆标注车。'])
    diagram.text(34, 1354, 'IoU 衡量两个框的重叠。核对标注时比较「预测和真车」；NMS 去重时比较「预测和预测」。', 19)
    diagram.save('flow.svg')


ROLES = [
    # 每行依次说明功能、直观用途和技术操作，层号与前向钩子的输出对应。
    ('基础特征提取', '从像素中提取局部线索，例如一小段边缘或一处颜色变化。', '3×3 卷积每次移动 2 个像素，生成 32 张特征图。'),
    ('下采样与特征提取', '把邻近的基础线索放在一起看，每个位置能看到更大一片图。', '步长 2 卷积把网格从 320×320 缩到 160×160。'),
    ('局部特征提取', '在同样密的网格上，用更多通道描述局部纹理和形状。', '分支卷积后合并，通道从 64 增至 128。'),
    ('下采样至细网格', '为后面定位小目标准备一张较密的网格。', '步长 2 卷积得到 80×80；相邻格点在输入图上相隔 8 像素。'),
    ('细节特征提取', '保留目标在什么位置的线索，第 15 层会把这份细节接回来。', '在 80×80 网格上做分支卷积，输出 256 个通道。'),
    ('下采样至中等网格', '把邻近细节汇到一起，让后面的层看更大范围的形状。', '步长 2 卷积把 80×80 缩到 40×40。'),
    ('中尺度特征提取', '在中等网格上组合形状线索，第 12 层会再次用到它们。', '分支卷积并合并，保留 40×40 网格和 256 个通道。'),
    ('下采样至粗网格', '把更大一片区域的信息汇到每个位置，供深层利用。', '步长 2 卷积得到 20×20；相邻格点在输入图上相隔 32 像素。'),
    ('深层特征提取', '把较大范围内的局部形状组合起来，为判断物体类别提供线索。', '在 20×20 网格上做分支卷积，再合并结果。'),
    ('多尺度特征汇集', '看同一位置时，既参考近处，也参考更大一圈的环境。', '连续三次 5×5 最大池化，拼接池化前后四路特征。'),
    ('空间注意力', '让一个位置也能参考图中较远位置的线索。', '部分通道做自注意力，按位置间的关联加权汇总，再与其余通道融合。'),
    ('上采样对齐', '把深层线索铺到更密的网格，让它和第 6 层逐格对齐。', '最近邻上采样复制已有数值，20×20 放大到 40×40。'),
    ('跨层特征拼接', '让后续卷积能同时读到深层的环境信息和第 6 层的形状线索。', '两路 40×40 特征按通道叠放，512 + 256 = 768 个通道。'),
    ('中尺度特征融合', '学习哪些深层线索和中层线索放在一起最有用。', '卷积混合第 12 层拼接的特征，压到 256 个通道。'),
    ('上采样对齐', '把已经融合的线索铺到细网格，准备接回第 4 层的位置细节。', '最近邻上采样把 40×40 放大到 80×80。'),
    ('接回位置细节', '让每个细格既能看到第 4 层的位置细节，也能读到深层线索。', '沿通道拼接两张 80×80 特征图，256 + 256 = 512 个通道。'),
    ('细网格特征融合', '给检测头提供最密的定位线索；本例框到小车的候选来自这一路。', '卷积融合出 P3，送往第 23 层检测头，也送往第 17 层。'),
    ('细节传回中尺度', '把 P3 的细节汇到中等网格，供下一层和中尺度线索对齐。', '步长 2 卷积把 80×80 缩到 40×40。'),
    ('跨层特征拼接', '让中等网格也能用到刚从细网格传回来的信息。', '拼接第 17 层与第 13 层，128 + 256 = 384 个通道。'),
    ('中网格特征融合', '给检测头提供兼有细节和周边信息的中等网格。', '卷积融合出 P4，送往第 23 层检测头，也送往第 20 层。'),
    ('特征传回粗尺度', '把 P4 的信息汇到粗网格，准备和深层信息放在一起。', '步长 2 卷积把 40×40 缩到 20×20。'),
    ('跨层特征拼接', '让粗网格同时保留从 P4 传来的线索和第 10 层的环境信息。', '拼接第 20 层与第 10 层，256 + 512 = 768 个通道。'),
    ('粗网格特征融合', '给检测头提供粗网格上的特征，帮助判断占图较大范围的目标。', '卷积融合出 P5，与 P3、P4 一起送入第 23 层。'),
    ('定位与分类预测', '在每个格点回答两个问题：物体框在哪里？它更像哪一类？', '三路特征分别预测四边距离和十类分数，共产生 8400 个候选。'),
]


def layers():
    # 用检查点本身前向，读取真实连接与张量，避免误用通用 80 类或其他模型尺度。
    net = YOLO(WEIGHT).model.eval()
    diagram = Diagram(1340, 2295, 'YOLO11s 第 0 至 23 层逐层输入、输出与作用')
    diagram.text(32, 52, '跟着一张 640 × 640 的图，走过 YOLO11s 的每一层', 33, True)
    diagram.text(32, 89, '每行对应权重里一个编号模块。形状按「通道 × 高 × 宽」阅读，省略批次；不同来源以层号标出。', 19)
    diagram.text(32, 121, '例如 128×80×80 表示 128 张各有 80 行、80 列的特征图。每个数是模型学到的响应强度。', 18)
    for left, title in [(32, '模块 / 用途'), (211, '输入来自'), (339, '输入形状'), (520, '输出形状'), (710, '这一层帮模型做什么？')]:
        diagram.text(left, 169, title, 19, True)

    def record(module, inputs, output):
        index = module.i
        top = 188 + index * 71
        sources = module.f if isinstance(module.f, list) else [module.f]
        source_labels = [str(index - 1 if item == -1 else item) for item in sources] if index else ['图片']
        incoming = inputs[0] if isinstance(inputs[0], list) else [inputs[0]]
        shape_text = lambda tensor: '×'.join(str(size) for size in tensor.shape[1:])
        diagram.rect(24, top, 1292, 65, BLUE if index < 11 else (GREEN if index < 23 else AMBER), 5)
        purpose, explanation, operation = ROLES[index]
        diagram.text(35, top + 26, f'{index:02}  {type(module).__name__}', 20, True)
        diagram.text(35, top + 52, purpose, 17, True, '#245e6d')
        diagram.text(211, top + 38, ' + '.join(source_labels), 18)
        for offset, tensor in enumerate(incoming):
            baseline = top + 38 + (offset - (len(incoming) - 1) / 2) * 20
            diagram.text(339, baseline, shape_text(tensor), 17)
        shape = shape_text(output) if index < 23 else shape_text(output[0]) + '（解码）'
        diagram.text(520, top + 38, shape, 18, True)
        diagram.text(710, top + 26, explanation, 17, True)
        diagram.text(710, top + 52, operation, 16)

    for module in net.model:
        module.register_forward_hook(record)
    with torch.inference_mode():
        net(torch.zeros(1, 3, 640, 640))
    diagram.card(24, 1907, 1292, 165, AMBER, '把第 23 层打开　｜　三张特征图 → 每格两个分支 → 一张候选表', [
        '第 16 层 P3 贡献 6400 格，第 19 层 P4 贡献 1600 格，第 22 层 P5 贡献 400 格，按此顺序排成 8400 列。',
        '位置分支输出 [B,64,8400]：左上右下四边各有 16 档。分类分支输出 [B,10,8400]：每个位置对应 VisDrone 十类。',
        '推理时对各边的 16 档加权求距离，再结合格点和步长解码；接上十个分类分数，得到 [B,14,8400]。B 是图片数。'])
    diagram.card(24, 2087, 1292, 176, '#e5eaef', '在上表找到这辆车　｜　第 16 层 → 第 23 层 → 候选 1633', [
        'P3 的网格间距为 8 像素。输入里 8.33 像素宽的车约跨一个格距；每格的预测也会用到周围特征。',
        'P3 第 20 行、第 33 列（从 0 起）对应 20 × 80 + 33 = 1633。此列解码框的 IoU 是 0.5186，最高类别是 people。',
        '这能定位到分类选择时已无合格车框。要解释前面哪层特征导致误判，还需要单独的干预实验。'])
    diagram.save('layers.svg')


if __name__ == '__main__':
    overview()
    layers()
