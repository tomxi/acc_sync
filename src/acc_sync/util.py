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
