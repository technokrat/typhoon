from collections import Counter

import numpy as np

from typhoon.helper import (
    add_residual_half_cycles,
    counter_to_full_interval_df,
    merge_cycle_counters,
)


def test_merge_cycle_counters():
    c1 = {(0.0, 1.0): 1, (1.0, 2.0): 2}
    c2 = {(0.0, 1.0): 3, (2.0, 3.0): 4}

    merged = merge_cycle_counters([c1, Counter(c2)])
    assert merged == {(0.0, 1.0): 4, (1.0, 2.0): 2, (2.0, 3.0): 4}


def test_add_residual_half_cycles():
    base = {(0.0, 1.0): 2}
    residual = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    updated = add_residual_half_cycles(base, residual)

    # base cycles stay the same
    assert updated[(0.0, 1.0)] == 2

    # residual half-cycles: (1,2) and (2,3) each add 0.5
    assert updated[(1.0, 2.0)] == 0.5
    assert updated[(2.0, 3.0)] == 0.5


def test_counter_to_full_interval_df():
    counter = {(0.1, 0.1): 5, (0.1, 0.2): 6, (0.2, 0.1): 7}
    bin_size = 0.1

    df = counter_to_full_interval_df(counter, bin_size)

    # Basic shape checks: MultiIndex with two levels
    assert df.index.nlevels == 2

    from_levels, to_levels = (
        df.index.levels  # pyright: ignore[reportAttributeAccessIssue]
    )

    # Check a few specific cells by explicit interval positions
    assert df.loc[(from_levels[0], to_levels[0]), "value"] == 5

    # Ensure zeros are present where no cycles exist
    zero_pos = (from_levels[1], to_levels[1])
    assert df.loc[zero_pos, "value"] == 0
