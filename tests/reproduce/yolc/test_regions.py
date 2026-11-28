import numpy as np

from reproduce.yolc.regions import cluster_windows, relocate


def test_cluster_and_matched_control():
    # 两个相隔的高分块；随机组只能移动裁片，不能改变分辨率或前向次数。
    heat = np.zeros((80, 128), dtype='f')
    heat[12:28, 14:40] = .9
    heat[54:72, 88:106] = .8
    windows = cluster_windows(heat, 1280, 800)
    assert len(windows) == 2
    shifted = relocate(windows, 1280, 800, 17)
    assert windows != shifted
    assert shifted == relocate(windows, 1280, 800, 17)
    assert all(0 <= left < right <= 1280 and 0 <= top < bottom <= 800 for left, top, right, bottom in shifted)
    for original, moved in zip(windows, shifted, strict=True):
        assert moved[2] - moved[0] == original[2] - original[0]
        assert moved[3] - moved[1] == original[3] - original[1]
    for center_x, center_y in [(270, 200), (970, 630)]:
        assert any(left <= center_x < right and top <= center_y < bottom for left, top, right, bottom in windows)
