from pathlib import Path


import pandas as pd
import numpy as np

def load_session_data(
    pt: str,
    session: str,
    roles: list[str],
    hands: list[str],
) -> dict[str, pd.DataFrame]:
    """Load parquet recordings and return as a dictionary of DataFrames."""
    data_dir = Path(__file__).parent.parent.parent / "data"
    
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
    """Trim a DataFrame to a specific time range."""

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

    return df.truncate(before=start_ts, after=end_ts)


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
        mag = trim_time(df['magnitude'], start_time, end_time)
        # Fill missing and filter
        mag = sig_filter(fill_missing(mag), cutoff=filter_cutoff, fs=25, btype='lowpass')
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


def stft(signal: pd.Series, hop: float = 0.5, win_length: float = 5.0, sr: float = 25.0, verbose: bool = False):
    from librosa import stft as librosa_stft
    window_length = int(round(win_length * sr))
    # find nearest power of 2
    n_fft = int(2 ** int(np.ceil(np.log2(window_length))))
    hop_length = int(round(hop * sr))

    if verbose:
        print(f"STFT parameters: n_fft={n_fft}, hop_length={hop_length}, win_length={window_length}")
    # STFT: complex-valued matrix, shape = (frequency_bins, time_frames)
    return librosa_stft(
        y=signal.values.astype(float),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=window_length,
        center=True
    )


def spec_coherence(S_i, S_j, sigma=(1.0, 2.0), eps=1e-8):
    """
    Computes the smoothed magnitude-squared coherence between two spectrograms.
    
    Args:
        S_i, S_j: 2D complex arrays (freq_bins, time_frames).
        sigma: Standard deviation for Gaussian smoothing (freq_smooth, time_smooth).
               Often, smoothing over time is prioritized over frequency.
        eps: Small constant for numerical stability.
        
    Returns:
        coherence: 2D real array bounded [0, 1].
    """
    from scipy.ndimage import gaussian_filter
    # 1. Compute raw spectra
    cross_spec = S_i * S_j.conj()
    auto_i = np.abs(S_i)**2
    auto_j = np.abs(S_j)**2

    # 2. Apply smoothing (Note: Cross-spectrum must be smoothed in real/imaginary parts separately)
    smooth_cross_real = gaussian_filter(cross_spec.real, sigma=sigma)
    smooth_cross_imag = gaussian_filter(cross_spec.imag, sigma=sigma)
    smooth_cross = smooth_cross_real + 1j * smooth_cross_imag

    smooth_auto_i = gaussian_filter(auto_i, sigma=sigma)
    smooth_auto_j = gaussian_filter(auto_j, sigma=sigma)

    # 3. Compute magnitude-squared coherence
    numerator = np.abs(smooth_cross)**2
    denominator = smooth_auto_i * smooth_auto_j

    return numerator / (denominator + eps)

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