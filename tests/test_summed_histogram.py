from typhoon import summed_histogram


def test_summed_histogram_decreasing_and_cumulative():
    # keys intentionally unordered to verify sorting in decreasing order
    hist = {
        1.0: 2.0,
        3.0: 1.0,
        2.0: 3.0,
    }

    result = summed_histogram(hist)

    # Expect processing in decreasing order of s_a,ers: 3.0, 2.0, 1.0
    expected = [
        (3.0, 1.0),  # 1
        (2.0, 4.0),  # 1 + 3
        (1.0, 6.0),  # 1 + 3 + 2
    ]

    assert isinstance(result, list)
    assert result == expected
