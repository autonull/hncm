import copylm


def test_match_labels_point_to_exact_spans() -> None:
    data = b"q\nabcXYZabcXYZ\n"
    src, ln = copylm.match_labels(data, 64, 6, 8, 4)
    i = 8
    assert src[i] == 2
    assert data[src[i] : src[i] + ln[i]] == data[i : i + ln[i]]
    for t in range(len(data)):
        if src[t] >= 1:
            assert data[src[t] : src[t] + ln[t]] == data[t : t + ln[t]]


def test_buckets() -> None:
    assert copylm._bucket(8) == 0
    assert copylm._bucket(15) == 0
    assert copylm._bucket(16) == 1
    assert copylm._bucket(64) == 3
