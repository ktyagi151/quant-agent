"""Point-in-time membership reconstruction — unit tested against a synthetic changes table."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from quant_agent import universe as uni


@pytest.fixture
def synthetic_changes():
    """3 synthetic changes, 3 synthetic current members."""
    changes = pd.DataFrame(
        {
            "effective_date": pd.to_datetime(
                ["2020-06-15", "2021-03-10", "2022-09-01"]
            ),
            # AAA joined on 2020-06-15 (was NOT member before).
            # BBB joined on 2021-03-10 replacing OLD1 (OLD1 was member before, BBB was not).
            # CCC joined on 2022-09-01 replacing OLD2.
            "added": ["AAA", "BBB", "CCC"],
            "removed": [None, "OLD1", "OLD2"],
        }
    )
    current = ["AAA", "BBB", "CCC", "ALWAYS"]  # ALWAYS never changed
    return changes, current


def test_reconstruction_correctness(synthetic_changes):
    changes, current = synthetic_changes

    with patch.object(uni, "get_sp500_tickers", return_value=current), patch.object(
        uni, "get_membership_changes", return_value=changes
    ):
        mem = uni.build_membership_matrix("2020-01-01", "2023-01-01")

    # ALWAYS is a member throughout.
    assert mem["ALWAYS"].all()

    # AAA: not a member before 2020-06-15, is after.
    assert not mem.loc["2020-03-02", "AAA"]
    assert mem.loc["2020-06-15", "AAA"]
    assert mem.loc["2020-12-01", "AAA"]

    # BBB: joined 2021-03-10. Before that, NOT a member.
    assert not mem.loc["2021-01-04", "BBB"]
    assert mem.loc["2021-03-10", "BBB"]
    # OLD1: was a member before 2021-03-10, not after.
    assert mem.loc["2021-01-04", "OLD1"]
    assert not mem.loc["2021-03-10", "OLD1"]
    assert not mem.loc["2022-06-01", "OLD1"]

    # CCC: joined 2022-09-01. OLD2 was a member before.
    assert not mem.loc["2022-06-01", "CCC"]
    assert mem.loc["2022-09-01", "CCC"]
    assert mem.loc["2022-06-01", "OLD2"]
    assert not mem.loc["2022-09-01", "OLD2"]


def test_matrix_is_bool_and_has_business_days(synthetic_changes):
    changes, current = synthetic_changes
    with patch.object(uni, "get_sp500_tickers", return_value=current), patch.object(
        uni, "get_membership_changes", return_value=changes
    ):
        mem = uni.build_membership_matrix("2021-01-01", "2021-01-31")
    assert mem.dtypes.iloc[0] == bool
    # All business days in Jan 2021.
    expected = pd.bdate_range("2021-01-01", "2021-01-31")
    assert list(mem.index) == list(expected)
