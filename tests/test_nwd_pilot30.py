import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_p1_small_nwd import qualifies


@pytest.mark.parametrize("ap,recall,expected", [
    (.30, 0., True), (.299, 0., False), (.30, -.001, False),
    (-.1, 1., True), (-.100001, 1.5, False), (0., .9999, False),
    (-.1746, 1.5464, False), (math.nan, 2., False), (1., math.inf, False),
])
def test_full30_followup_preserves_ap_guardrail_and_two_distinct_gain_routes(ap, recall, expected):
    assert qualifies({"mAP50_95": ap, "small_recall": recall}) is expected
