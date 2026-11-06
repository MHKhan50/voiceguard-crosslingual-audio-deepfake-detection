from voiceguard.protocol import parse_line, read_protocol, stratified_sample


def test_parse_lines(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("LA_0079 LA_T_1138215 - - bonafide\nLA_0079 LA_T_1271820 - A01 spoof\nbad line\n")
    items = read_protocol(str(p))
    assert [(u.attack, u.label) for u in items] == [("bonafide", 0), ("A01", 1)]
    assert parse_line("x y z") is None


def test_stratified_sample_is_reproducible_and_capped():
    lines = ["S%d U%03d - A07 spoof" % (i % 5, i) for i in range(50)]
    lines += ["S%d B%03d - - bonafide" % (i % 5, i) for i in range(10)]
    items = [parse_line(l) for l in lines]
    a = stratified_sample(items, {"bonafide": 4}, 8, seed=1)
    b = stratified_sample(items, {"bonafide": 4}, 8, seed=1)
    assert a == b
    assert sum(u.attack == "A07" for u in a) == 8
    assert sum(u.attack == "bonafide" for u in a) == 4
