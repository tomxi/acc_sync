import pandas as pd
import numpy as np
import librosa
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _naive_time(index) -> pd.DatetimeIndex:
    """Return a tz-naive DatetimeIndex so every panel shares one wall-clock axis."""
    idx = pd.DatetimeIndex(index)
    return idx.tz_localize(None) if idx.tz is not None else idx


# region: MOI visualization
def plot_session_notes(data: pd.DataFrame, date: str = "2026-01-01") -> go.Figure:
    """Plot session notes as horizontal bars on a session-adjusted time axis.

    Pass ``date`` as the recording date so the notes align with the signal panels.
    """
    # Convert event offsets into actual session times.
    session_start = pd.to_timedelta(data["Video Start (HH:MM:SS)"].dropna().iloc[0])
    start = pd.Timestamp(date) + session_start + pd.to_timedelta("00:" + data["Event Start (MM:SS)"])
    end = pd.Timestamp(date) + session_start + pd.to_timedelta("00:" + data["Event End (MM:SS)"])

    # Combine all note columns into one hover label per row.
    notes = data.iloc[:, 3:].apply(
        lambda row: "<br>".join(str(n).strip() for n in row if pd.notna(n) and str(n).strip()),
        axis=1,
    )

    fig = px.timeline(
        pd.DataFrame({"start": start, "end": end, "lane": "Notes", "notes": notes}),
        x_start="start", x_end="end", y="lane", hover_name="notes",
        hover_data={"lane": False},
    )
    fig.update_layout(title="Session Notes", xaxis_title="Time", yaxis_title="", showlegend=False)
    return fig

# endregion

# region: Accelerometer visualization
def plot_acc(data: pd.DataFrame) -> go.Figure:
    """Plot X, Y, Z and magnitude accelerometer channels as overlaid lines."""
    x = _naive_time(data.index)
    fig = go.Figure()
    for col in ["x", "y", "z", "magnitude"]:
        fig.add_scatter(x=x, y=data[col].values, mode="lines", name=col)
    fig.update_layout(title="Accelerometer Data", xaxis_title="Time", yaxis_title="Acceleration")
    return fig


def plot_mag_dict(mag_dict: dict[str, pd.Series]) -> go.Figure:
    """Overlay each wearable's magnitude series on a shared time axis."""
    fig = go.Figure()
    for name, series in mag_dict.items():
        fig.add_scatter(x=_naive_time(series.index), y=series.values, mode="lines", name=name)
    fig.update_layout(title="Session ACC Magnitude", xaxis_title="Time", yaxis_title="Magnitude")
    return fig

# endregion

# region: Correlation visualization
def plot_corr_df(corr_df: pd.DataFrame) -> go.Figure:
    """Heatmap of windowed cross-correlation: time on x, lag on y."""
    fig = go.Figure(go.Heatmap(
        x=_naive_time(corr_df.index),
        y=corr_df.columns.astype(float),
        z=corr_df.T.values,
        colorscale="Inferno",
        colorbar=dict(title="r"),
    ))
    fig.update_layout(title="Windowed Cross Correlation", xaxis_title="Time", yaxis_title="Lag (s)")
    return fig


def plot_line(
    series: pd.Series,
    name: str | None = None,
    title: str | None = None,
    yaxis_title: str | None = None,
    yrange: list[float] | None = None,
) -> go.Figure:
    """Line plot of a time-indexed Series, ready to drop into ``combine_time_figs``.

    ``name`` sets the legend label (defaults to the Series' name).
    """
    name = name or getattr(series, "name", None)
    fig = go.Figure(go.Scatter(
        x=_naive_time(series.index), y=series.values, mode="lines",
        name=name, showlegend=name is not None,
    ))
    fig.update_layout(
        title=title, xaxis_title="Time", yaxis=dict(title=yaxis_title, range=yrange),
    )
    return fig

# endregion

# region: Spectrogram visualization
def plot_spec(S: pd.DataFrame) -> go.Figure:
    """Heatmap of an STFT spectrogram (dB) with frequency on y and time on x.

    ``S`` is a complex spectrogram (frequency index, datetime columns) from ``util.stft``.
    """
    fig = go.Figure(go.Heatmap(
        x=_naive_time(S.columns),
        y=S.index.astype(float),
        z=librosa.amplitude_to_db(np.abs(S.values), ref=np.max),
        colorscale="Magma",
        colorbar=dict(title="dB"),
    ))
    fig.update_layout(title="Spectrogram", xaxis_title="Time", yaxis_title="Frequency (Hz)")
    return fig


def plot_coherence(coherence_matrix: pd.DataFrame) -> go.Figure:
    """Heatmap of a magnitude-squared coherence matrix bounded to [0, 1].

    ``coherence_matrix`` has a frequency index and datetime columns (from
    ``util.spec_coherence`` given DataFrame inputs).
    """
    fig = go.Figure(go.Heatmap(
        x=_naive_time(coherence_matrix.columns),
        y=coherence_matrix.index.astype(float),
        z=coherence_matrix.values,
        colorscale="Inferno",
        zmin=0.0, zmax=1.0,
        colorbar=dict(title="Coherence"),
    ))
    fig.update_layout(
        title="Magnitude-Squared Coherence", xaxis_title="Time", yaxis_title="Frequency (Hz)",
    )
    return fig

# endregion

# region: Composition
def combine_time_figs(
    figs: list[go.Figure],
    titles: list[str] | None = None,
    row_heights: list[float] | None = None,
    row_height_px: int = 250,
) -> go.Figure:
    """Stack single-panel, time-axis figures into one shared-x-axis figure.

    Each input figure becomes one row. Heatmap colorbars are repositioned to
    align with their own row so they don't overlap.
    """
    figs = list(figs)
    n = len(figs)
    combined = make_subplots(
        rows=n, cols=1, shared_xaxes=True,
        vertical_spacing=min(0.05, 0.8 / max(n - 1, 1)),
        row_heights=row_heights,
        subplot_titles=titles,
    )

    for row, src in enumerate(figs, start=1):
        for trace in src.data:
            combined.add_trace(trace, row=row, col=1)
        src_yaxis = src.layout.yaxis
        combined.update_yaxes(
            title_text=src_yaxis.title.text, range=src_yaxis.range, row=row, col=1,
        )

    # Align each heatmap's colorbar with its subplot's vertical domain.
    for row in range(1, n + 1):
        axis_name = "yaxis" if row == 1 else f"yaxis{row}"
        lo, hi = combined.layout[axis_name].domain
        for trace in combined.select_traces(row=row, col=1):
            if isinstance(trace, go.Heatmap):
                trace.update(colorbar=dict(
                    len=hi - lo, y=(lo + hi) / 2, yanchor="middle", thickness=12,
                ))

    # Force a date axis so px.timeline bars (which store start in `base` and
    # width in `x`) are positioned correctly instead of on an inferred linear axis.
    combined.update_xaxes(type="date")
    combined.update_xaxes(title_text="Time", row=n, col=1)
    combined.update_layout(height=row_height_px * n, margin=dict(t=40, r=90))
    return combined

# endregion