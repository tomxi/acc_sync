import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import librosa

def plot_acc(data: pd.DataFrame, figsize: tuple[float, float]=(8, 6)) -> plt.Figure:
    """
    Plot accelerometer data in 1 column with 4 subplots.
    Plots X, Y, Z, and the magnitude (euclidean norm) of the vector.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with columns 'x', 'y', 'z', and 'magnitude'
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True, sharey=True)
    data['x'].plot(ax=axes[0])
    data['y'].plot(ax=axes[1])
    data['z'].plot(ax=axes[2])
    data['magnitude'].plot(ax=axes[3])

    # label the x-axis
    axes[3].set_xlabel('Time')

    # Label the subplots
    axes[0].set_ylabel('X')
    axes[1].set_ylabel('Y')
    axes[2].set_ylabel('Z')
    axes[3].set_ylabel('Magnitude')

    # label the figure
    fig.suptitle('Accelerometer Data')
    return fig

def plot_mag_dict(mag_dict: dict[str, pd.Series], figsize: tuple[float, float]=(8, 6)) -> plt.Figure:
    """
    Plot multiple magnitude series on the same figure in a single column.
    
    Parameters:
    -----------
    mag_dict : dict[str, pd.Series]
        Dictionary mapping names to magnitude series
    """
    num_rows = len(mag_dict)
    fig, axes = plt.subplots(num_rows, 1, figsize=figsize, sharex=True, sharey=True)
    for i, (name, series) in enumerate(mag_dict.items()):
        series.plot(ax=axes[i])
        axes[i].set_ylabel(name)
    axes[-1].set_xlabel('Time')
    fig.suptitle('Session ACC Magnitude')
    return fig

def plot_corr_df(corr_df: pd.DataFrame, figsize=(8, 6)) -> plt.Figure:
    """
    Plot a correlation DataFrame with time on the x-axis and lags on the y-axis.
    Then the max for each time point in the second subplot.
    
    Parameters:
    -----------
    corr_df : pd.DataFrame
        DataFrame with time on the index and lags as columns
    """
    fig, axes = plt.subplots(
        2, 1, figsize=figsize, sharex=True, height_ratios=[2, 1]
    )

    time_index = pd.DatetimeIndex(corr_df.index)
    if time_index.tz is not None:
        time_index = time_index.tz_localize(None)
    x = mdates.date2num(time_index.to_pydatetime())
    y = corr_df.columns.astype(float)

    axes[0].pcolormesh(
        x,
        y,
        corr_df.T.values,
        shading="auto",
        cmap="inferno",
    )
    axes[0].set_ylabel('Lag (s)')
    axes[0].set_title('Windowed Cross Correlation with Different Lags')

    axes[1].plot(x, corr_df.max(axis=1).values)
    axes[1].set_ylabel('Max Cross Correlation')
    axes[1].set_xlabel('Time')
    axes[1].set_ylim([0, 1])

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(formatter)
    
    return fig

def plot_spec(
    S: np.ndarray, sr: float = 25.0, hop_length: int = 12, n_fft: int = 128, 
    figsize: tuple[float, float]=(10, 4), ax=None
) -> plt.Axes:
    """
    Plot a spectrogram.
    
    Parameters:
    -----------
    S : np.ndarray
        Spectrogram
    sr : float
        Sampling rate
    hop_length : int
        Hop length
    n_fft : int
        FFT size
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(S),ref=np.max),
        sr=sr, hop_length=hop_length, n_fft=n_fft, 
        x_axis="time", y_axis="hz", ax=ax)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    return ax

def plot_coherence(coherence_matrix, sr, hop_length, y_axis='linear', ax=None):
    """
    Plots a bounded coherence matrix with correct physical time/frequency axes.
    
    Args:
        coherence_matrix: 2D array (freq_bins, time_frames) bounded [0, 1].
        sr: Sampling rate.
        hop_length: STFT hop length.
        y_axis: Frequency scale ('linear' or 'log').
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        colorbar = True
    else:
        fig = ax.get_figure()
        colorbar = False
    
    # Render the array mapping indices to physical units
    img = librosa.display.specshow(
        coherence_matrix,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis=y_axis, 
        ax=ax,
        cmap='inferno', # Perceptually uniform, excellent for highlighting localized structure
        vmin=0.0,     # Strictly enforce coherence bounds
        vmax=1.0
    )
    
    ax.set_title('STFT Magnitude-Squared Coherence')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')
    
    if colorbar:
        # Attach a colorbar mapped to the [0, 1] bounds
        cbar = fig.colorbar(img, ax=ax, format="%.1f")
        cbar.set_label('Coherence $\gamma^2$')
        plt.tight_layout()
    return fig, ax