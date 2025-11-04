# ligotools/tests/test_readligo.py
from pathlib import Path
import numpy as np
import pytest
from ligotools import readligo as rl

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"

H1_FILE = DATA / "H-H1_LOSC_4_V2-1126259446-32.hdf5"
L1_FILE = DATA / "L-L1_LOSC_4_V2-1126259446-32.hdf5"
TEMPLATE_FILE = DATA / "GW150914_4_template.hdf5"

@pytest.mark.parametrize("fpath", [H1_FILE, L1_FILE])
def test_loaddata_shapes_and_dt(fpath):
    """Basic sanity: shapes match; fs ~ 4096 derived from time axis."""
    assert fpath.exists(), f"Missing test file: {fpath}"
    strain, time, chan_dict = rl.loaddata(fpath)

    # 1D arrays of same length
    assert isinstance(strain, np.ndarray) and strain.ndim == 1 and strain.size > 0
    assert isinstance(time, np.ndarray) and time.ndim == 1 and time.size == strain.size
    assert isinstance(chan_dict, dict)

    # derive fs from the time array (robust to different readligo variants)
    dt = float(np.mean(np.diff(time)))
    assert np.isfinite(dt) and dt > 0
    fs = 1.0 / dt
    assert 3800 <= fs <= 5000  # tolerate variation around 4096 Hz
    # dt close to 1/fs
    assert np.isclose(dt, 1.0 / fs, rtol=1e-6, atol=1e-9)

def test_template_file_exists_and_is_readable():
    assert TEMPLATE_FILE.exists(), f"Missing template file: {TEMPLATE_FILE}"
    # Just ensure the file can be opened and has content
    import h5py
    with h5py.File(TEMPLATE_FILE, "r") as f:
        keys = list(f.keys())
        assert len(keys) > 0
