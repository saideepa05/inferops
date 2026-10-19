import pytest

from bench.quality import extract_answer, is_correct, wilson_interval


@pytest.mark.parametrize("text, expected", [
    ("3 + 4 = 7 apples. The answer is 7.", 7),
    ("Total is $1,250. The answer is: $1,250", 1250),
    ("The answer is **18**", 18),
    ("She keeps 16 - 3 - 4 = 9 eggs, so 9 * 2 = 18", 18),  # no phrase: last number
    ("The answer is -2.5", -2.5),
    ("I don't know.", None),
])
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


def test_is_correct_normalizes_gold_formatting():
    assert is_correct(1250.0, "1,250")
    assert not is_correct(None, "3")
    assert not is_correct(7.0, "8")


def test_wilson_interval_contains_point_estimate_and_narrows_with_n():
    lo, hi = wilson_interval(160, 200)
    assert lo < 0.8 < hi
    assert 0.73 < lo < 0.75 and 0.84 < hi < 0.86
    lo_small, hi_small = wilson_interval(40, 50)
    assert hi_small - lo_small > hi - lo
