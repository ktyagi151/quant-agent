from __future__ import annotations

import numpy as np
import pandas as pd

from quant_agent import features as feat


def test_mom_12_1_known(synthetic_panel):
    m = feat.mom_12_1(synthetic_panel)
    # First 252 rows must be NaN; after that, non-NaN.
    assert m.iloc[:252].isna().all().all()
    assert m.iloc[252:].notna().all().all()


def test_reversal_5d_matches_definition(synthetic_panel):
    r = feat.reversal_5d(synthetic_panel)
    px = synthetic_panel["adj_close"]
    expected = -(px / px.shift(5) - 1)
    pd.testing.assert_frame_equal(r, expected)


def test_volume_shock_zero_when_constant(synthetic_panel):
    panel = {**synthetic_panel}
    panel["volume"] = pd.DataFrame(
        1_000_000.0, index=panel["volume"].index, columns=panel["volume"].columns
    )
    vs = feat.volume_shock(panel)
    # After 20-day warmup, log(1/1) = 0.
    assert np.allclose(vs.iloc[20:].values, 0.0)


def test_compute_features_registry(synthetic_panel):
    out = feat.compute_features(synthetic_panel, ["mom_12_1", "reversal_5d"])
    assert set(out.keys()) == {"mom_12_1", "reversal_5d"}
    assert out["mom_12_1"].shape == synthetic_panel["adj_close"].shape
