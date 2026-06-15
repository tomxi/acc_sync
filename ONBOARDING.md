# Onboarding Guide (WIP)

## Setting up your environment

If you feel confident with Python and package management, you can skip this section.

### Setting up a Conda environment 
To avoid dependency conflicts, it's recommended to use a virtual environment for this project.
We'll use conda to manage our environment.

If you don't have conda or miniconda installed, download and install the latest miniconda by running the appropriate installer for your operating system from [miniconda.com](https://www.anaconda.com/docs/getting-started/miniconda/install/overview).

Once you have miniconda or conda installed, you can create a virtual environment for this project by running this in your terminal:
```bash
conda create -n acc_sync python=3.12
```
Feel free to replace `acc_sync` with whatever name you prefer for your virtual environment. We use python 3.12 so we stay away from the bleeding edge.

When prompted, confirm the installation by typing `y`.

### Activating the Conda environment
After the environment is created, you need to activate it every time you want to work on this project.

We do so by running:
```bash
conda activate acc_sync
```

Everything we do from now on will be done with this environment active.

When you're done working on this project, you can deactivate the environment by running:
```bash
conda deactivate
```

### Setting up Jupyter (optional)

If you wish to use jupyter notebook or jupyter lab, we can install them in this environment.
With the environment activated, run:
```bash
conda install jupyterlab notebook ipykernel
```

You can then launch jupyter lab by running:
```bash
jupyter lab
```

[Trouble Shooting]
If you wish to use run Jupyter notebooks in other IDEs, like VS Code or Cursor, you should be able to see this environment already.
However, if your IDE doesn't show this environment, you can make this environment availabe as a named Jupyter kernel by running:
```bash
python -m ipykernel install --user --name acc_sync --display-name "Python (acc_sync)"
```
You can then select this kernel in your IDE when working with Jupyter notebooks.

## Cloning and installing the project
With the envirnoment set up, we can now clone and install the project.

First, navigate to the directory where you want to clone the repository:
```bash
cd /path/to/your/code/directory
```

Now, clone the repository by running:
```bash
git clone https://github.com/tomxi/acc_sync.git
cd acc_sync
```

Make sure you are working in your desired environment, then install the project and its dependencies by running:
```bash
pip install -e .
```

This will install the project in "editable" mode, which means that any changes you make to the source code will be reflected in the installed package, perfect for development.

## First Steps
Now, you should be able to run the `demo.ipynb` notebook.

If you have been following this guide, you can skip the first cell of the notebook, which is intended for Google Colab and sets up the environment.

There are some sample data included in the repository for running the demo notebook, but you should populate the `data/` directory with more data to be analyzed.

## Ingesting Manual Coding Data
Input notes to a spreadsheet: you can use this [google sheets template](https://docs.google.com/spreadsheets/d/1PgzVLJpMRNzUZVZKHbXM6U6CJG_ndnuM7JzruOdDtyU/edit?usp=sharing) and make a copy.

Once you have your data in the spreadsheet, you can export it as a CSV file and place it in the corresponding session folder in the `data/` directory.

Then you should be able to ingest and visualize the notes using the following code:
```python
import acc_sync as acs
import pandas as pd

# Change the path to where your csv file is stored
notes_df = pd.read_csv("data/RF001/2025_12_13/session_notes.csv")
fig = acs.viz.plot_session_notes(notes_df, date="2025-12-13")
fig.show()
```
