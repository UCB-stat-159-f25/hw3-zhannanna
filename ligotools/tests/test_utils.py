from pathlib import Path
import numpy as np
import pytest
from ligotools.utils import whiten, reqshift, write_wavfile, plot_psd_panel

def test_whiten_accepts_callable_psd_and_changes_scale():
    fs = 4096.0
    dt = 1.0 / fs
    t = np.arange(4096) * dt
    x = np.sin(2 * np.pi * 1000 * t)
    psd_callable = lambda f: np.ones_like(f)
    xw = whiten(x, psd_callable, dt)
    assert xw.shape == x.shape
    assert np.all(np.isfinite(xw))
    assert 1e-6 < np.var(xw) < 100.0

def test_reqshift_preserves_length_and_power_reasonably():
    fs = 4096.0
    dt = 1.0 / fs
    t = np.arange(4096) * dt
    x = np.sin(2 * np.pi * 400 * t)
    y = reqshift(x, fshift=200, sample_rate=fs)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    assert 0.1 < np.var(y) / np.var(x) < 10.0

def test_write_wavfile_roundtrip(tmp_path: Path):
    fs = 16000
    t = np.arange(1600) / fs
    x = 0.5 * np.sin(2 * np.pi * 440 * t)
    out = tmp_path / "test_tone.wav"
    write_wavfile(out, fs, x)
    from scipy.io import wavfile as _wav
    rfs, data = _wav.read(out)
    assert rfs == fs
    assert data.ndim == 1 and data.size == x.size
    assert data.dtype == np.int16
    assert np.max(np.abs(data)) <= 32767

def test_plot_psd_panel_writes_png(tmp_path: Path):
    fs = 4096
    t = np.arange(4096) / fs
    noise = np.random.randn(t.size)
    out = tmp_path / "asd_noise.png"
    ret = plot_psd_panel(noise, fs=fs, outpath=out)
    assert ret is None
    assert out.exists() and out.stat().st_size > 0
