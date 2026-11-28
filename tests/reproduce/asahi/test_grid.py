from reproduce.asahi.grid import grid


def test_grid_coverage_and_orientation():
    # 验证阈值两侧、横竖图以及非整数步长的边界覆盖。
    for width, height, count in ((1818, 720, 6), (1819, 720, 12), (720, 1920, 12), (960, 540, 6)):
        tiles = grid(width, height)
        assert len(set(tiles)) == count
        for axis, length in ((0, width), (1, height)):
            spans = sorted({(box[axis], box[axis+2]) for box in tiles})
            assert spans[0][0] == 0 and spans[-1][1] == length
            assert all(second[0] < first[1] for first, second in zip(spans, spans[1:]))
