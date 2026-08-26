from paleopy.utils import flush_msg, is_nan, fix_encoding, tqdm


def test_is_nan_true_for_nan_string():
    assert is_nan(float("nan")) is True


def test_is_nan_false_for_normal_values():
    assert is_nan(0) is False
    assert is_nan("hello") is False
    assert is_nan(None) is False


def test_flush_msg_prints_carriage_return(capsys):
    flush_msg("hello")
    captured = capsys.readouterr()
    assert captured.out == "\rhello"


def test_tqdm_fallback_or_real_is_iterable():
    result = list(tqdm([1, 2, 3]))
    assert result == [1, 2, 3]


def test_fix_encoding_leaves_plain_ascii_unchanged():
    assert fix_encoding("plain ascii") == "plain ascii"
