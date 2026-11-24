"""检验两种评测入口共享的 VisDrone 忽略区与真值处理。"""

import cv2
import numpy as np
import pytest

from scripts.reproduce.esod.evaluate_visdrone import prepare_image


@pytest.mark.parametrize("masked", [False, True])
def test_author_validation_pixels_and_boxes(tmp_path, masked):
    # 纹理图可以检出忽略区涂灰后是否多做或漏做了一次 JPEG 编解码。
    path = tmp_path / "sample.jpg"
    annotation = tmp_path / "sample.txt"
    texture = np.random.default_rng(42).integers(0, 256, (64, 80, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), texture)
    expected = cv2.imread(str(path))
    rows = ["5,6,7,8,1,10,1,2", "4,4,4,4,1,1,2,0"]
    if masked:
        rows.extend(["0,0,12,12,0,0,0,0,", "24,8,9,8,0,11,0,0"])
        expected[:12, :12] = 85
        expected[8:16, 24:33] = 85
        disk = tmp_path / "sample_masked.jpg"
        assert cv2.imwrite(str(disk), expected)
        expected = cv2.imread(str(disk))
    annotation.write_text("\n".join(rows), encoding="utf-8")
    actual, boxes = prepare_image(path, annotation)
    np.testing.assert_array_equal(actual, expected)
    # 遮挡等级 2 仍进入真值，截断等级 2 被丢弃，类别编号保持原始 1–10。
    assert boxes == [(10, [5, 6, 7, 8])]
