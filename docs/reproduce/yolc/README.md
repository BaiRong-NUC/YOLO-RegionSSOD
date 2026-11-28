# YOLC 局部放大复现卡

| 论文基线 CenterNet | 论文方法 YOLC |
|---|---|
| ![CenterNet 整图检测](centernet-baseline-architecture.svg) | ![YOLC 密集区局部放大](yolc-architecture.svg) |

## YOLO11s 上的等预算选区对照

**要分开两个问题：多看几个裁片有没有用，以及选对裁片是否比随机选更好。** 以下是 YOLC 选区机制的迁移实验，不是用 YOLO11s 重训作者的 CenterNet 模型。原模型的权重评测和独立训练结果单列在[本机对照](#本机对照)。

| YOLO11s 权重 | 推理方式 | 548 图额外裁片总数 | 总体 AP | 小目标 AP | 平均 ms/原图 |
|---|---|---:|---:|---:|---:|
| 0 | 仅整图 | 0 | 16.093 | 6.784 | 8.57 |
| 0 | 整图 + 随机位置 | 1011 | 18.023 | 9.508 | 53.92 |
| 0 | 整图 + 密集区 | 1011 | 24.632 | 17.231 | 52.62 |
| 1 | 仅整图 | 0 | 16.392 | 6.904 | 14.37 |
| 1 | 整图 + 随机位置 | 999 | 18.172 | 9.544 | 46.48 |
| 1 | 整图 + 密集区 | 999 | 24.723 | 17.210 | 46.48 |

逐图核对，两臂在两份权重上分别使用完全相同的裁片数。密集区相对随机位置的**小目标 AP 提高 7.723 / 7.666 点**（从未舍入值计算）。随机裁片本身也优于只看整图，但两种裁片策略都比只看整图慢。这个结果支持「在给定前向预算下，重检位置影响精度」；并未证明它优于等预算均匀网格，或一定能在新权重、独立划分上复现。两份历史 `best.pt` 都用过 val 选模。

- **整图**：640×640 输入只跑一次 YOLO11s。**密集区**：从整图 P3 的十类最大预测分数取热图，去补边并映射到原图。沿用作者的 16×10 网格、前 15 个密集格、八连通聚类，最多选两个包围区域；滤掉面积小于 96² 的区域，边长外扩至 1.2 倍。每个裁片也输入 640×640；不用作者 CenterNet 的热图头和 1.5 倍图像插值。
- **随机位置**：先走同一选区流程，再把每个窗口平移到图内随机位置。随机数种子由图像采样像素的 CRC32 决定，不使用标注。与密集区臂逐图匹配窗口数量、宽高、模型前向次数和输入尺寸；位置带来的窗口重叠或目标覆盖差异未消除。两臂都用局部框替换完整落在窗口内的整图框。随机臂只有一套位置抽样，不代表所有随机网格。
- **评测**：同一批 VisDrone val 548 图，同一 RTX 4070 Ti，两份已训练 YOLO11s 权重依次正序、反序运行三臂；置信度 0.001、IoU 0.7、每图最多 500 框。原图尺度的 COCO 小目标 AP 以点计。耗时含选区、检测、结果合并和回传，不含读盘；每个权重每臂一次计时。[环境、权重与预处理](../README.md#本轮配对协议)。

```powershell
reproduce/sahi/local/env/Scripts/python.exe -m pytest tests/reproduce/yolc
reproduce/sahi/local/env/Scripts/python.exe -m scripts.reproduce.yolc.evaluate_regions --split val
```

第一个命令是合成样本测试；第二个命令生成 `reproduce/yolc/local/val-548.json`，包含每臂检查点 SHA-256、逐图窗口数、耗时和精度指标。历史作者模型结果使用另一个文件 `val-pair.json`。

## 本机对照

**局部放大提高小目标 AP 的效果在小样本独立训练中未稳定复现。** 同一份作者权重两次都提高 2.81 点。换成本机独立训练的两份权重，小目标 AP 分别下降 0.01685 点和提高 0.00013 点。两个种子的小目标召回率各提高约半点，每张原图都多花约 140 毫秒。两份独立权重的整图小目标 AP 均低于 0.13 点，这轮只能说明小样本条件下的表现。全量独立训练和原始 CenterNet 对照仍待验证。

问题是，在同一张航拍图、同一个检测器上，额外放大最多两处目标密集的地方，能否提高小目标检测精度？每对实验共享一份 YOLC 权重。整图组（k=0）保留整图检测，局部组（k=2）再跑最多两个裁片。这样能测出局部重检带来的变化。图中左侧的原始 CenterNet 是论文整模型对照，尚缺同协议的独立基线检查点。

**权重评测复现。** 在全部 548 张图上开启局部放大，两轮的小目标 AP 均从 27.83 升到 30.64（+2.81 点），小目标平均召回率从 47.08 升到 51.71（+4.63 点）。两轮平均每张原图分别多花 178、134 毫秒。这两轮复用了同一份公开权重。两图工程烟测仅验证代码能够运行。原始正式结果见 `reproduce/yolc/local/val-pair.json`。

| 轮次 | 检测方式 | AP | AP50 | 小目标 AP | 小目标 AR | 平均毫秒/图 | P95 毫秒/图 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 整图 k=0 | 36.07 | 60.07 | 27.83 | 47.08 | 165.97 | 187.57 |
| 1 | 局部放大 k=2 | 37.80 | 61.99 | 30.64 | 51.71 | 343.80 | 505.97 |
| 2 | 局部放大 k=2 | 37.80 | 61.99 | 30.64 | 51.71 | 316.66 | 395.03 |
| 2 | 整图 k=0 | 36.07 | 60.07 | 27.83 | 47.08 | 182.72 | 262.40 |

## 实现与协议

**独立训练复现（16 张训练原图的小样本初筛）。** 表中 AP 和 AR 均按 0–100 计。两个种子开启局部放大时，都在全部 548 张验证图上实际触发了局部检测，各重检 975 个裁片。召回率上升说明可以找回少量小目标，AP 也取决于检测框的位置和置信度排序。当前结果未区分二者对 AP 的影响。

| 训练种子 | 推理方式 | 总体 AP | 小目标 AP | 小目标 AR | 平均毫秒/原图 |
|---|---|---:|---:|---:|---:|
| 0 | 整图 | 0.09911 | 0.12388 | 3.513 | 165.44 |
| 0 | 局部放大 | 0.08751 | 0.10703 | 3.976 | 308.33 |
| 1 | 整图 | 0.00265 | 0.00485 | 0.437 | 166.37 |
| 1 | 局部放大 | 0.00374 | 0.00497 | 0.952 | 305.12 |

### 小样本独立训练协议

从排序后的 VisDrone train 原图中用固定抽样种子 2026 抽 16 张，使用原始训练标注和作者的 `gen_crop.py` 得到 16 张原图、16 张密集区裁片，共 992 个训练框。两个种子共享这批图，只载入 ImageNet 预训练的 HRNet-W48 骨干，检测头分别初始化。每组训练 30 轮，batch 2，训练输入 512×320，数据集重复次数 1，初始学习率 0.00125，预热 20 步。固定取第 30 轮权重，用作者的单尺度原图评测流程在全部 548 张验证图上配对比较。种子 0 先整图后局部，种子 1 先局部后整图。主指标为小目标 AP，另记录小目标 AR、总体 AP、每张原图平均耗时和真正触发局部检测的图数。

原始评测记录位于 `reproduce/yolc/local/pilot16/seed{0,1}/val-pair.json`。两个 `epoch_30.pth` 的 SHA-256 分别为 `bfd94ecb4edfaa63374d0ad91523dc17e86b2387c2bc243642359070ce4e135b` 和 `831c341d83c2ca2dfdbe854085882a785e40bcaf8202320a71e2744b4efae268`。

在根目录运行（训练依赖 `tqdm==4.67.1`）：

```powershell
$python = 'reproduce/yolc/local/env/python.exe'
& $python -m pip install tqdm==4.67.1
curl.exe -fL -o reproduce/yolc/local/hrnetv2_w48-d2186c55.pth https://download.openmmlab.com/pretrain/third_party/hrnetv2_w48-d2186c55.pth
foreach ($seed in 0,1) {
    & $python -B scripts/reproduce/yolc/train_pilot.py --seed $seed
    if ($LASTEXITCODE) { throw "种子 $seed 训练失败" }
    $run = "reproduce/yolc/local/pilot16/seed$seed"
    $order = if ($seed -eq 0) { 0,2 } else { 2,0 }
    & $python -B scripts/reproduce/yolc/evaluate.py --checkpoint "$run/epoch_30.pth" --order $order --output "$run/val-pair.json"
    if ($LASTEXITCODE) { throw "种子 $seed 评测失败" }
}
```

### 作者权重评测协议

- **论文与权重。** Liu 等，IEEE T-ITS 2024，DOI `10.1109/TITS.2024.3386928`，核对 arXiv `2404.06180v2` 表 I。作者提交 `3b42257ca18d212f7c24ed82eaa7e050d3d6e33d` 提供训练、评测、配置和验证标注；检查点从作者 README 的 Google Drive 下载，SHA-256 `7ad105b019cce54a6004f37be5adaa041d2334cda2e8fe684063df8ce81af6b9`。Zotero 分类为 `Research/YOLO-RegionSSOD/无人机水平框小目标检测/YOLC`。[论文](https://arxiv.org/abs/2404.06180v2) · [作者提交与权重入口](https://github.com/dawn-ech/YOLC/tree/3b42257ca18d212f7c24ed82eaa7e050d3d6e33d)
- **文献报告。** 表 I 在 VisDrone val 上给出的原始 CenterNet、YOLC k=0、YOLC k=2 的 AP 分别为 29.1、35.5、37.8，小目标 AP 分别为 22.3、26.9、30.5。这里使用单尺度原图的行。判断本机收益时比较两个本机组。
- **数据与模型。** 使用作者 COCO 格式标注中的 548 张 VisDrone2019-DET val 图像，输入原图比例 1.0，无翻转。模型为 HRNet-W48、HRFPN 和 YOLC 检测头，预测十类水平框。k=2 按作者实现将区域外扩 1.2 倍、裁片放大 1.5 倍，保留区域外的原图检测框。
- **评测与计时。** 取 COCO IoU 0.50～0.95、每类每图上限 1000 的 AP、小目标 AP 与小目标 AR。忽略全零占位框。每组预热三图、GPU 同步，计入作者入口的读图、聚类、检测、合并时间，关闭热图调试文件输出。默认按 0、2、2、0 跑两轮。重复运行同一权重只检验评测重复性，跨训练种子的稳定性需另测。

## 运行入口

本机环境为 Windows、RTX 4070 Ti 12 GB、Python 3.10、PyTorch 1.13.1+cu117、torchvision 0.14.1+cu117、MMCV-full 1.7.0、MMDetection 2.26.0、Kornia 0.6.9、NumPy 1.26.4、pycocotools 2.0.11。将作者检查点放在 `reproduce/yolc/local/yolc.pth`，原图放在 `data/visdrone/images/val/`。在仓库根目录执行以下命令，结果文件位于 Git 忽略目录内。

首次配置 Windows x64 环境：

```powershell
git submodule update --init reproduce/yolc/upstream
conda create -y -p "$PWD/reproduce/yolc/local/env" python=3.10 pip
$python = 'reproduce/yolc/local/env/python.exe'
& $python -m pip install 'torch==1.13.1+cu117' 'torchvision==0.14.1+cu117' --extra-index-url https://download.pytorch.org/whl/cu117 'numpy==1.26.4' 'pillow==10.4.0'
& $python -m pip install 'https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/mmcv_full-1.7.0-cp310-cp310-win_amd64.whl' 'mmdet==2.26.0' 'kornia==0.6.9' 'pycocotools==2.0.11' 'opencv-python==4.11.0.86' 'yapf==0.40.1' 'setuptools<80'
```

烟测、全量配对评测与绘图：

```powershell
$python = 'reproduce/yolc/local/env/python.exe'
& $python -B scripts/reproduce/yolc/evaluate.py --limit 2 --order 0 2 --output reproduce/yolc/local/smoke.json
& $python -B scripts/reproduce/yolc/evaluate.py --output reproduce/yolc/local/val-pair.json
python reproduce/yolc/figures/draw_pair.py
```
