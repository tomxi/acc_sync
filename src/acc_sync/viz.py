import matplotlib.pyplot as plt

def plot_acc(data):
    """
    Plot accelerometer data in 1 column with 4 subplots.
    Plots X, Y, Z, and the magnitude (euclidean norm) of the vector.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with columns 'x', 'y', 'z', and 'magnitude'
    """
    fig, axes = plt.subplots(4, 1, figsize=(8, 6), sharex=True, sharey=True)
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