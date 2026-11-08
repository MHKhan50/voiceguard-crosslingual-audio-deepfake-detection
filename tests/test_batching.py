from voiceguard.batching import equal_length_batches


def test_batches_hold_equal_lengths_and_cover_every_clip_once():
    lengths = [64000, 30000, 64000, 30000, 12345, 64000, 64000]
    batches = equal_length_batches(lengths, batch_size=3)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    assert all(len({lengths[i] for i in b}) == 1 for b in batches)
    assert max(len(b) for b in batches) == 3
    assert [len(b) for b in batches] == [3, 1, 2, 1]
