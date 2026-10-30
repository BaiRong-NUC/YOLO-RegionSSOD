from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_p1_no_soft_weights import continuation


def test_followup_requires_paired_overall_and_small_object_evidence():
    assert continuation({"mAP50_95": .3, "small_recall": 0})
    assert continuation({"mAP50_95": -.1, "small_recall": 1.0})
    assert not continuation({"mAP50_95": .29, "small_recall": .99})
    assert not continuation({"mAP50_95": .5, "small_recall": -.1})
    assert not continuation({"mAP50_95": -.11, "small_recall": 1.1})
