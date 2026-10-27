from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from run_p1_scale_seed_pair import check_epochs, parameter_differences
from scale_seed_tables import METRICS, paired_statistics


@pytest.mark.parametrize("epochs", [list(range(1, 30)), list(range(2, 32)), [1] * 30])
def test_acceptance_rejects_incomplete_or_repeated_epochs(tmp_path, epochs):
    (tmp_path / "results.csv").write_text("epoch,time\n" + "".join(f"{e},1\n" for e in epochs))
    with pytest.raises(RuntimeError, match="continuous epochs"):
        check_epochs(tmp_path)


def test_pairing_does_not_hide_seed_or_loss_changes():
    before = {"seed": 1, "data": "scale", "batch": 8, "cls": .5}
    after = {"seed": 2, "data": "global", "batch": 8, "cls": .6}
    assert parameter_differences(before, after, {"data", "name", "save_dir"}) == {
        "seed": [1, 2], "cls": [.5, .6]}


def test_paired_sd_uses_within_seed_differences():
    # Perfectly correlated arms: large between-seed SD, constant paired effect.
    pairs = {str(i): {"scale_aware": dict.fromkeys(METRICS, a),
                     "global_t050": dict.fromkeys(METRICS, b),
                     "delta_global_minus_scale_aware_pp": dict.fromkeys(METRICS, -10.)}
             for i, (a, b) in enumerate(((.4, .3), (.6, .5)))}
    aggregate, delta = paired_statistics(pairs)
    assert aggregate["scale_aware"]["mAP50"]["mean_percent"] == 50
    assert aggregate["scale_aware"]["mAP50"]["sample_std_pp"] == pytest.approx(2 ** .5 * 10)
    assert delta["mAP50"] == {"mean_pp": -10, "sample_std_pp": 0}
