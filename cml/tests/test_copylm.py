import numpy as np

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


def test_match_len_and_exact_copies() -> None:
    w = b"zabcdefghabcdefgh"
    assert copylm._match_len(w, 1, 9, 8) == 8
    assert copylm._match_len(w, 9, 1, 8) == 0
    idx = np.array([8])
    pred = np.array([0])  # key 0 means source byte 1
    run = np.array([0])  # bucket 0 means length 8
    assert copylm._exact_copies(w, idx, pred, run, np.array([True]))[0]
    assert not copylm._exact_copies(w, idx, pred, run, np.array([False]))[0]


def test_valid_sources_accept_every_exact_match() -> None:
    w = b"abcXYZabcQQQabc"
    idx = np.array([len(w) - 4])
    mask = copylm._valid_sources(w, idx, 3, 32)
    at = int(idx[0]) + 1
    slots = np.nonzero(mask[0])[0]
    assert slots.tolist() == [5]
    for j in slots:
        assert w[j + 1 : j + 1 + 3] == w[at : at + 3]
