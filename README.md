# Accelerometer Synchrony
Measuring movement synchrony between subjects wearing Bangle.js 2 smartwatches.

## Setup

Navigate to the project directory and run:
```bash
pip install -e .
```

There's some sample data in the `data/` directory, but you'll need to put the actual parquet files there.

## Usage

```python
import acc_sync as acs
import matplotlib.pyplot as plt

session = acs.util.load_session_data(
    pt="RF001",
    session="2025_11_15",
    roles=["client", "therapist"],
    hands=["R", "L"]
)

acc_mag = acs.util.prep_session_mag(
    session, start_time='15:03:00', end_time='15:32:00'
)

for wearable in session:
    fig = acs.viz.plot_acc(session[wearable])
    fig.suptitle(f"Accelerometer Data for {wearable}")
    plt.show()

fig = acs.viz.plot_mag_dict(acc_mag, figsize=(8, 6))
plt.show()
```

## Demo

See `demo.ipynb` for a demo of the package.
GitHub sometimes have trouble converting notebooks, so you can also view (and run) it on [Google Colab](https://colab.research.google.com/github/tomxi/acc_sync/blob/main/demo.ipynb).