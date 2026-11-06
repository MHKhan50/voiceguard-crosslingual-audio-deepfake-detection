import numpy as np

from voiceguard.channel import apply_condition, mu_law


def _tone(f, sr=16000, sec=1.0):
    t = np.arange(int(sr * sec)) / sr
    return 0.5 * np.sin(2 * np.pi * f * t)


def _power_at(x, f, sr=16000):
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    return spec[np.argmin(np.abs(freqs - f))]


def test_phone_keeps_length_and_removes_high_band():
    x = _tone(1000) + _tone(6000)
    y = apply_condition(x, "phone")
    assert len(y) == len(x) and y.dtype == np.float32
    assert _power_at(y, 6000) < 1e-3 * _power_at(y, 1000)


def test_mu_law_is_close_but_lossy():
    x = np.linspace(-0.9, 0.9, 1001)
    y = mu_law(x)
    assert np.max(np.abs(y - x)) < 0.05 and not np.allclose(x, y)


def test_noise_condition_is_seeded():
    x = _tone(500)
    a = apply_condition(x, "phone_noise15", seed=3)
    b = apply_condition(x, "phone_noise15", seed=3)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, apply_condition(x, "phone"))
