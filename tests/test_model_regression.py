"""Model regression tests on a FROZEN test set: the model must beat the reference baseline with
statistical support, stay under a committed loss ceiling, and the frozen set must not change."""
import numpy as np
import pytest

from dstk.leakage import frame_fingerprint
from dstk.stats import bootstrap_ci, paired_comparison


def _losses(hooks):
    X, y = hooks.get_frozen_test()
    model = hooks.load_pipeline()
    return (np.asarray(hooks.per_item_loss(y, model.predict(X)), float),
            np.asarray(hooks.per_item_loss(y, hooks.baseline_predict(X)), float))


# ----------------------------------------------------------------------------------- hooks
def test_frozen_test_set_has_not_changed(hooks, require):
    require("get_frozen_test", "expected_frozen_fingerprint")
    X, y = hooks.get_frozen_test()
    assert frame_fingerprint(X.assign(__y__=np.asarray(y))) == hooks.expected_frozen_fingerprint(), \
        "frozen test set content changed; regenerate the committed hash only on purpose"


def test_model_beats_baseline_with_statistical_support(hooks, require):
    require("get_frozen_test", "load_pipeline", "baseline_predict", "per_item_loss", "regression_config")
    model_loss, base_loss = _losses(hooks)
    r = paired_comparison(model_loss, base_loss)
    assert r["verdict"] == "A better", f"model is not clearly better than the baseline: {r}"
    gain = -r["median_diff"]
    assert gain >= hooks.regression_config().get("min_median_gain", 0.0), \
        f"median per-item gain over baseline is only {gain:.4f}"


def test_mean_loss_stays_below_committed_ceiling(hooks, require):
    require("get_frozen_test", "load_pipeline", "baseline_predict", "per_item_loss", "regression_config")
    model_loss, _ = _losses(hooks)
    mean, lo, hi = bootstrap_ci(model_loss, np.mean, n_boot=2000)
    ceiling = hooks.regression_config()["max_mean_loss"]
    assert mean <= ceiling, f"mean loss {mean:.4f} (95% CI {lo:.4f}-{hi:.4f}) exceeds ceiling {ceiling}"


def test_worst_decile_is_reported_and_finite(hooks, require):
    require("get_frozen_test", "load_pipeline", "baseline_predict", "per_item_loss")
    model_loss, _ = _losses(hooks)
    worst = np.sort(model_loss)[-max(1, len(model_loss) // 10):]
    assert np.isfinite(worst).all()
    print(f"worst-decile mean loss: {worst.mean():.4f} (overall mean {model_loss.mean():.4f})")


# ----------------------------------------------------------------------------------- self-tests
def test_gate_rejects_a_model_that_is_not_better_than_baseline():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.1, 0.4, 80)
    same = base + rng.normal(0, 0.001, 80)
    assert paired_comparison(same, base)["verdict"] == "no clear difference"
    worse = base + 0.1
    assert paired_comparison(worse, base)["verdict"] == "B better"


def test_gate_accepts_a_clearly_better_model():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.2, 0.4, 80)
    better = base - rng.uniform(0.05, 0.12, 80)
    assert paired_comparison(better, base)["verdict"] == "A better"
