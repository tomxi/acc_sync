from pathlib import Path


import pandas as pd

def load_session_data(
    pt: str,
    session: str,
    role: str,
    hand: str,
) -> pd.DataFrame:
    """Load a single parquet recording and return as DataFrame."""
    data_dir = Path(__file__).parent.parent.parent / "data"

    path = data_dir / pt / session / f"{pt}_{role}_{hand}_25Hz.parquet"
    df = pd.read_parquet(path)
    return df


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