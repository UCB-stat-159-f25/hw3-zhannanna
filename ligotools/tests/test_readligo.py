# ligotools/tests/test_readligo.py
from pathlib import Path
import numpy as np
import pytest

# The tutorial usually imports like: from ligotools import readligo as rl
from ligotools import readligo as rl

# Locate the repo root from this file, then the data folder
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"

H1_FILE = DATA / "H-H1_LOSC_4_V2-1126259446-32.hdf5"
L1_FILE = DATA / "L-L1_LOSC_4_V2-1126259446-32.hdf5"
TEMPLATE_FILE = DATA / "GW150914_4_template.hdf5"

@pytest.mark.parametrize("fpath", [H1_FILE, L1_FILE])
def test_loaddata_shapes_and_dt(fpath):
    """Basic sanity: shapes match, fs around 4kHz, dt ~= 1/fs."""
    assert fpath.exists(), f"Missing test file: {fpath}"
    strain, time, chan_dict = rl.loaddata(fpath)

    # 1D arrays of same length
    assert isinstance(strain, np.ndarray) and strain.ndim == 1 and strain.size > 0
    assert isinstance(time, np.ndarray) and time.ndim == 1 and time.size == strain.size

    # sampling frequency in metadata
    assert isinstance(chan_dict, dict) and "fs" in chan_dict
    fs = float(chan_dict["fs"])
    assert 3800 <= fs <= 5000  # be a bit tolerant across data variants

    # dt ~ 1/fs
    dt = np.mean(np.diff(time))
    assert np.isfinite(dt)
    assert np.isclose(dt, 1.0 / fs, rtol=1e-3, atol=1e-6)

def test_template_loadable():
    """We can access the template file via readligo helper if present, else via h5py."""
    assert TEMPLATE_FILE.exists(), f"Missing template file: {TEMPLATE_FILE}"
    if hasattr(rl, "read_template"):
        template, t = rl.read_template(TEMPLATE_FILE)
        assert template.ndim == 1 and t.ndim == 1 and template.size == t.size and template.size > 100
    else:
        # Fall back: just confirm the file opens and has content.
        import h5py
        with h5py.File(TEMPLATE_FILE, "r") as f:
            # Many tutorials store a dataset named 'template' or similar.
            # We only require the file to be readable and non-empty.
            keys = list(f.keys())
            assert len(keys) > 0
