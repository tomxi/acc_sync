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
    viz.plot_acc(session[wearable]).update_layout(
        title=f"Accelerometer Data: {wearable}"
    ).show()

viz.plot_mag_dict(acc_mag).show()
```
