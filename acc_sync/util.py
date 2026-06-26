import warnings
from pathlib import Path


import pandas as pd
import numpy as np

# Accelerometer counts corresponding to 1 g. The raw signal baseline sits around
# this value (gravity), so it serves as a stable, session-independent physical
# reference for reporting spectral energies in dB.
COUNTS_PER_G = 1000.0

def load_session_data(
    pt: str,
    session: str,
    roles: list[str],
    hands: list[str],
) -> dict[str, pd.DataFrame]:
    """Load parquet recordings and return as a dictionary of DataFrames."""
    data_dir = Path(__file__).parent.parent / "data"
    
    df_dict = dict()
    for role in roles:
        for hand in hands:
            path = data_dir / pt / session / f"{pt}_{role}_{hand}_25Hz.parquet"
            # make sure file exists
            if not path.exists():
                raise FileNotFoundError(f"File {path} does not exist")
            df_dict[f"{role}_{hand}"] = pd.read_parquet(path)
    return df_dict


def trim_time(
    df: pd.DataFrame,
    start_time: str = '15:06:00', 
    end_time: str = '15:34:19'
) -> pd.DataFrame:
    """Resample a Series/DataFrame onto a uniform grid spanning [start, end].

    Rather than a plain truncation (which keeps each recording's own irregular
    samples and can yield different lengths), the data is reindexed onto a uniform
    time grid at the inferred sampling period, covering exactly [start_time,
    end_time]. This guarantees consistent sample positions and length across
    recordings so downstream STFT/correlation frames line up.

    If the requested range extends beyond the available data (nothing before
    ``start_time`` or nothing after ``end_time``), a warning is raised and the
    extrapolated ticks are filled with NaN.
    """
    # Make sure df index is DatetimeIndex and monotonic increasing
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # Get the date from the first index, and check recording spans only one day
    date = df.index[0].date()
    if df.index[-1].date() != date:
        raise ValueError("Start and end times must be on the same day")

    # Convert start and end times to timestamps
    start_ts = pd.Timestamp(f"{date} {start_time}")
    end_ts = pd.Timestamp(f"{date} {end_time}")

    # Localize timestamps to the same timezone as the index
    tz = df.index.tz
    if tz is not None:
        start_ts = start_ts.tz_localize(tz)
        end_ts = end_ts.tz_localize(tz)

    # Warn (and pad with NaN below) if the requested range exceeds the data.
    if start_ts < df.index[0]:
        warnings.warn(
            f"No data before {df.index[0]}; extrapolating ticks from {start_ts} "
            "and filling with NaN.",
            stacklevel=2,
        )
    if end_ts > df.index[-1]:
        warnings.warn(
            f"No data after {df.index[-1]}; extrapolating ticks to {end_ts} "
            "and filling with NaN.",
            stacklevel=2,
        )

    # Build a uniform grid at the data's sampling period and snap samples onto it
    # (nearest within half a period absorbs jitter). Ticks with no nearby sample
    # become NaN, keeping every recording the same length.
    period = pd.Timedelta(np.median(np.diff(df.index.values)))
    grid = pd.date_range(start=start_ts, end=end_ts, freq=period)
    return df.reindex(grid, method="nearest", tolerance=period / 2)


def fill_missing(series: pd.Series, method: str = 'linear') -> pd.Series:
    """
    Fill missing values in a series by using linear or spline interpolation. 
    
    Args:
        series: The series to fill.
        method: The method to use for filling. Can be 'linear' or 'spline'.
    
    Returns:
        The filled series.
    """
    # interpolate
    if method == 'linear':
        filled = series.interpolate(method='linear')
    elif method == 'spline':
        filled = series.interpolate(method='spline', order=3)
    else:
        raise ValueError("Method must be 'linear' or 'spline'")
    
    # fill any remaining NaNs with 0
    return filled.fillna(0)


def sig_filter(series: pd.Series, cutoff: float = 10.0, fs: float = 25.0, btype: str = 'lowpass') -> pd.Series:
    """
    Apply a 4th order butterworth filter to a series.
    
    Args:
        series: The series to filter.
        cutoff: The cutoff frequency in Hz.
        fs: The sampling frequency in Hz.
        btype: The type of filter to apply. Can be 'lowpass' or 'highpass'.
    
    Returns:
        The filtered series.
    """
    from scipy.signal import butter, sosfiltfilt
    
    # See if signal has NaNs; if so, raise a warning and use fill_missing to interpolate
    if series.isna().any():
        # import warnings
        # warnings.warn("Signal has NaNs; using fill_missing to interpolate")
        series = fill_missing(series)

    # Design butterworth filter
    nyquist = 0.5 * fs
    sos = butter(4, cutoff / nyquist, btype=btype, output='sos')
    
    # Apply filter
    filtered = sosfiltfilt(sos, series.values)
    
    return pd.Series(filtered, index=series.index)


def normalize(series: pd.Series, type: str = 'zscore') -> pd.Series:
    """
    Normalize a series by subtracting the mean and dividing by the standard deviation 
    (z-score),or by the range (min-max normalization).
    
    Args:
        series: The series to normalize.
        type: The type of normalization to apply. Can be 'zscore' or 'range'.
    
    Returns:
        The normalized series.
    """
    if type == 'zscore':
        return (series - series.mean()) / series.std()
    elif type == 'range':
        return (series - series.min()) / (series.max() - series.min())
    else:
        raise ValueError("Type must be 'zscore' or 'range'")

def prep_session_mag(
    df_dict: dict[str, pd.DataFrame], 
    start_time: str = '15:05:00', 
    end_time: str = '15:30:00',
    filter_cutoff: float = 10.0
) -> dict[str, pd.DataFrame]:
    """
    Prepare session data by trimming time, filling missing values, and filtering.
    
    Args:
        df_dict: A dictionary of DataFrames.
        start_time: The start time to trim the data.
        end_time: The end time to trim the data.
        filter_cutoff: The cutoff frequency for the filter in Hz.
    
    Returns:
        A dictionary of prepared DataFrames.
    """
    series_dict = {}
    for key, df in df_dict.items():
        # get magnitude series and trim time
        mag = fill_missing(trim_time(df['magnitude'], start_time, end_time))
        # Fill missing and filter
        mag = sig_filter(mag, cutoff=filter_cutoff, fs=25, btype='lowpass')
        # High pass filter
        mag = sig_filter(mag, cutoff=0.1, fs=25, btype='highpass')
        series_dict[key] = mag
    return series_dict

def frame_xcorr(
    x: pd.Series, 
    y: pd.Series, 
    fs: float = 25.0, 
    window: float = 5.0, 
    hop: float = 0.5, 
    max_lag: float = 2.0
) -> pd.DataFrame:
    """
    Frame two signals and compute the pearson correlation between corresponding frames.
    
    Args:
        x, y: The two signals to correlate.
        fs: The sampling frequency in Hz.
        window: The window duration in seconds.
        hop: The hop size in seconds.
    
    Returns:
        A pandas DataFrame containing the correlation values for each frame and lag in seconds.
        The index is the time in seconds, and the columns are the lags in seconds.
        Positive lag means y is delayed relative to x.
    """
    from librosa import frames_to_time
    from librosa.util import frame
    from scipy import stats

    frame_length = int(round(window * fs))
    hop_length = int(round(hop * fs))
    max_lag_samples = int(round(max_lag * fs))
    
    # Frame 1 signal first: each frame is a window of frame_length samples
    # librosa returns shape (frame_length, n_frames), advancing by hop_length
    x_frames = frame(x, frame_length=frame_length, hop_length=hop_length)
    # Time axis in seconds, marking the center of each frame
    times = frames_to_time(range(x_frames.shape[-1]), sr=fs, hop_length=hop_length, n_fft=frame_length)
    # find actual start time from input series and build new time index for output
    start_time = x.index[0]
    times = start_time.tz_localize(None) + pd.to_timedelta(times, unit="s")
    if start_time.tzinfo is not None:
        times = times.tz_localize(start_time.tzinfo)
    
    # identify all the lags and shift y accordingly, then compute frame correlation
    lags = range(-max_lag_samples, max_lag_samples + 1)
    corr = dict()
    for lag in lags:
        y_frames = frame(y.shift(lag).fillna(0), frame_length=frame_length, hop_length=hop_length)
        # Compute correlation for each frame
        corr[lag / fs] = pd.Series(
            stats.pearsonr(x_frames, y_frames, axis=0)[0], 
            index=times
        )
    
    return pd.DataFrame(corr)


def stft(signal: pd.Series, hop: float = 0.5, win_length: float = 5.0, sr: float = 25.0, verbose: bool = False) -> pd.DataFrame:
    """
    Compute the STFT of a signal and return it as a DataFrame.

    Returns:
        A complex-valued DataFrame of shape (frequency_bins, time_frames).
        The index is the frequency in Hz, and the columns are a DatetimeIndex
        marking the center time of each frame, anchored at ``signal.index[0]``.
        This datetime column axis matches the time axis produced by
        ``windowed_correlation``/``plot_corr_df`` so spectrograms can be plotted
        on a consistent time axis.
    """
    from librosa import stft as librosa_stft
    from librosa import fft_frequencies, frames_to_time

    window_length = int(round(win_length * sr))
    # find nearest power of 2
    n_fft = int(2 ** int(np.ceil(np.log2(window_length))))
    hop_length = int(round(hop * sr))

    if verbose:
        print(f"STFT parameters: n_fft={n_fft}, hop_length={hop_length}, win_length={window_length}")
    # STFT: complex-valued matrix, shape = (frequency_bins, time_frames)
    S = librosa_stft(
        y=signal.values.astype(float),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=window_length,
        center=True
    )

    # Frequency bins (Hz) for the index
    freqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # Frame center times (seconds), then anchor to the signal's start time
    frame_times = frames_to_time(range(S.shape[-1]), sr=sr, hop_length=hop_length, n_fft=n_fft)
    start_time = signal.index[0]
    times = start_time.tz_localize(None) + pd.to_timedelta(frame_times, unit="s")
    if start_time.tzinfo is not None:
        times = times.tz_localize(start_time.tzinfo)

    return pd.DataFrame(S, index=freqs, columns=times)


def spec_coherence(S_i, S_j, sigma=(1.0, 2.0), eps=1e-8):
    """
    Computes the smoothed magnitude-squared coherence between two spectrograms.
    
    Args:
        S_i, S_j: 2D complex arrays (freq_bins, time_frames).
        sigma: Standard deviation for Gaussian smoothing (freq_smooth, time_smooth).
               Often, smoothing over time is prioritized over frequency.
        eps: Small constant for numerical stability.
        
    Returns:
        coherence: 2D real array bounded [0, 1]. If the inputs are DataFrames
        (e.g. from ``stft``), the result preserves their index/columns.
    """
    from scipy.ndimage import gaussian_filter

    # Accept DataFrames (from stft) or raw ndarrays; operate on raw values.
    index = S_i.index if isinstance(S_i, pd.DataFrame) else None
    columns = S_i.columns if isinstance(S_i, pd.DataFrame) else None
    S_i = np.asarray(S_i)
    S_j = np.asarray(S_j)

    # Compute raw spectra
    cross_spec = S_i * S_j.conj()
    auto_i = np.abs(S_i)**2
    auto_j = np.abs(S_j)**2

    # Apply smoothing (Note: Cross-spectrum must be smoothed in real/imaginary parts separately)
    smooth_cross_real = gaussian_filter(cross_spec.real, sigma=sigma)
    smooth_cross_imag = gaussian_filter(cross_spec.imag, sigma=sigma)
    smooth_cross = smooth_cross_real + 1j * smooth_cross_imag

    smooth_auto_i = gaussian_filter(auto_i, sigma=sigma)
    smooth_auto_j = gaussian_filter(auto_j, sigma=sigma)

    # Compute magnitude-squared coherence
    numerator = np.abs(smooth_cross)**2
    denominator = smooth_auto_i * smooth_auto_j

    coherence = numerator / (denominator + eps)
    if index is not None:
        coherence = pd.DataFrame(coherence, index=index, columns=columns)
    return coherence


def mean_cross_energy(cross_spec: pd.DataFrame, ref_g: float = 1.0, db=True) -> pd.Series:
    """Mean cross-spectral energy per time frame, in dB relative to (ref_g * g)^2.

    The cross-spectrum ``S_i * conj(S_j)`` has units of counts^2, so its magnitude
    is an energy. We reference it to a fixed physical level ``(ref_g * COUNTS_PER_G)^2``
    rather than a per-session full scale (dBFS), which keeps values directly
    comparable across sessions.

    Args:
        cross_spec: Complex cross-spectrum DataFrame (freq_bins, time_frames) with
            a datetime column index, e.g. ``S_i * conj(S_j)``.
        ref_g: Reference acceleration in g (default 1 g, the gravity baseline).

    Returns:
        A Series indexed by the cross-spectrum's time columns, in dB.
    """
    from librosa import power_to_db

    mean_energy = np.abs(cross_spec).mean(axis=0)
    ref = (ref_g * COUNTS_PER_G) ** 2
    result = pd.Series(
        power_to_db(np.asarray(mean_energy, dtype=float), ref=ref),
        index=mean_energy.index,
    )
    if not db:
        result = 10 ** (result / 10)
    return result


def physical_to_bin_sigmas(sigma_sec, sigma_hz, sr, n_fft, hop_length):
    """
    Converts physical smoothing targets to STFT bin dimensions.
    
    Args:
        sigma_sec: Target temporal smoothing standard deviation in seconds.
        sigma_hz: Target spectral smoothing standard deviation in Hertz.
        sr: Audio sampling rate.
        n_fft: FFT size used in the STFT.
        hop_length: Hop length used in the STFT.
        
    Returns:
        Tuple (sigma_f, sigma_t) to be used directly in gaussian_filter.
    """
    # Calculate resolutions
    delta_t = hop_length / sr
    delta_f = sr / n_fft
    
    # Calculate bin sigmas
    sigma_t = sigma_sec / delta_t
    sigma_f = sigma_hz / delta_f
    
    # Return in (freq, time) order for 2D array filtering
    return (sigma_f, sigma_t)