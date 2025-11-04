# ligotools/utils.py
# Utilities used by the LIGO tutorial notebook:
#  - whiten: handle PSD given as interp1d (callable) or as an array
#  - reqshift: frequency-shift a real signal
#  - write_wavfile: safe WAV writer that creates parent dirs
#  - plot_psd_panel: compute+save ASD (Welch)

from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt


__all__ = ["whiten", "reqshift", "write_wavfile", "plot_psd_panel"]


def whiten(strain: np.ndarray, psd_or_interp, dt: float) -> np.ndarray:
    """
    Whiten a strain time series.

    Parameters
    ----------
    strain : 1D np.ndarray
        Time-domain data.
    psd_or_interp : callable or 1D array
        Either a callable (e.g., scipy.interpolate.interp1d) that returns PSD(f),
        or a 1D array of PSD values (will be interpolated to the rfftfreq grid).
        PSD must be *one-sided*.
    dt : float
        Sampling interval in seconds.

    Returns
    -------
    1D np.ndarray
        Whitened time series (same length as `strain`).
    """
    strain = np.asarray(strain, dtype=float)
    Nt = strain.size
    hf = np.fft.rfft(strain)
    freqs = np.fft.rfftfreq(Nt, dt)

    # Evaluate PSD on the frequency grid used by rfft
    if callable(psd_or_interp):
        psd_vals = psd_or_interp(freqs)
    else:
        psd_vals = np.asarray(psd_or_interp, dtype=float)
        if psd_vals.shape != freqs.shape:
            # Interpolate provided PSD array to the rfftfreq grid
            x_old = np.linspace(freqs[0], freqs[-1], psd_vals.size)
            psd_vals = np.interp(freqs, x_old, psd_vals)

    # Safety: avoid zeros/negatives which would blow up the division
    psd_vals = np.asarray(psd_vals, dtype=float)
    positive = psd_vals > 0
    if not np.any(positive):
        raise ValueError("PSD has no positive values.")
    psd_min = float(psd_vals[positive].min())
    psd_vals[~positive] = psd_min

    # One-sided PSD normalization: divide by sqrt(PSD / (dt/2))
    white_hf = hf / np.sqrt(psd_vals / (dt / 2.0))
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht


def reqshift(data: np.ndarray, fshift: float = 100.0, sample_rate: float = 4096.0) -> np.ndarray:
    """
    Frequency shift a real time-domain signal by fshift (Hz).
    Uses complex mixing in the frequency domain with bin rolling.

    Parameters
    ----------
    data : 1D np.ndarray
    fshift : float
        Shift in Hz (positive shifts up).
    sample_rate : float
        Sampling rate in Hz.

    Returns
    -------
    1D np.ndarray (real)
    """
    x = np.fft.rfft(np.asarray(data, dtype=float))
    T = len(data) / float(sample_rate)
    df = 1.0 / T
    nbins = int(np.round(fshift / df))

    # Roll real and imag parts by the same number of bins
    y = np.roll(x.real, nbins) + 1j * np.roll(x.imag, nbins)
    if nbins > 0:
        y[:nbins] = 0
    elif nbins < 0:
        y[nbins:] = 0

    z = np.fft.irfft(y, n=len(data))
    return np.real(z)


def write_wavfile(path: Path | str, fs: int, data: np.ndarray) -> None:
    """
    Write a mono WAV file at sample rate fs. Float data are scaled to int16.
    Creates parent directories if necessary.

    Parameters
    ----------
    path : str or Path
    fs   : int
    data : 1D array-like
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(data)
    if x.dtype.kind == "f":
        m = float(np.max(np.abs(x))) if x.size else 1.0
        y = (x / m * 32767.0).astype(np.int16) if m > 0 else np.zeros_like(x, dtype=np.int16)
    else:
        y = x.astype(np.int16)

    wavfile.write(p, int(fs), y)


def plot_psd_panel(
    strain: np.ndarray,
    fs: float,
    nperseg: int = 4 * 4096,
    noverlap: int | None = None,
    outpath: Path | str | None = None,
):
    """
    Compute and plot ASD via Welch. Saves to `outpath` if provided, otherwise returns the figure.

    Parameters
    ----------
    strain : 1D np.ndarray
    fs     : float
        Sampling rate (Hz).
    nperseg : int
        Welch segment length.
    noverlap : int or None
        Welch segment overlap (defaults to nperseg//2).
    outpath : str/Path or None
        If given, save PNG here and return None. If None, return matplotlib Figure.
    """
    strain = np.asarray(strain, dtype=float)
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, Pxx = signal.welch(
        strain, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant"
    )
    asd = np.sqrt(Pxx)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(freqs, asd)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("ASD [1/√Hz]")
    ax.set_title("Amplitude Spectral Density")
    ax.grid(True, which="both", ls=":")

    if outpath is not None:
        out = Path(outpath)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return None

    return fig
