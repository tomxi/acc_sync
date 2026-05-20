# Accelerometer Synchrony
Measuring movement synchrony between subjects wearing Bangle.js 2 smartwatches.

## Setup

Navigate to the project directory and run:
```bash
pip install -e .
```


## Usage

```python
import acc_sync as acs


data = acs.load_session_data(
    pt="RF001",
    session="2025_09_27",
    role="client",
    hand="R",
)

acs.trim_time(data, '12:06:00', '15:34:19')
```