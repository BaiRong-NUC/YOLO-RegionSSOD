import numpy as np

from reproduce.daasd.slicing import select_tier, starts


def test_density_thresholds():
    # 单元格阈值是严格大于；同一个单元最多计入一次每个框。
    for count, expected in ((0, 'low'), (15, 'low'), (16, 'medium'), (30, 'medium'), (31, 'high')):
        assert select_tier(np.zeros((count, 4)), 400, 300)[0] == expected
    cells = np.array([[left, top, left+10, top+10] for left in range(0, 400, 100) for top in range(0, 300, 100)])
    for count, expected in ((8, 'low'), (9, 'medium'), (15, 'medium'), (16, 'high')):
        assert select_tier(np.repeat(cells, count, axis=0), 400, 300)[0] == expected


def test_aligned_windows():
    assert starts(400, 640, 0.2) == [0]
    assert starts(1664, 640, 0.2) == [0, 512, 1024]
    assert starts(1700, 640, 0.2) == [0, 512, 1024, 1060]
