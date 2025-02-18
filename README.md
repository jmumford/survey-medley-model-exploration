# How to run notebooks

Generate a `.venv` using
```
uv sync
```

The `.venv` will be located in the root directory.  Simply use this when running the code or notebooks on your own.  With VSCode, simply select as the kernel and if using a Jupyter server, then launch with:
```
uv run --with jupyter jupyter lab
```

The main notebook is `survey_medley_rt_exploration.ipynb`.  The other notebooks are mostly code checks that are likely only useful for Jeanette.