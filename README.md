# Artifact Description

This is a replication package to reproduce Figure 2(f), Figure 2(h) (and their Supplementary
versions), Figure 6(g), Figure 6(h) and Figure 7(b) proposed in the paper.

## Requirements

### Hardware Requirements

Tested configuration:
- CPU: Intel Core i7-11800H (2.30 GHz)
- RAM: 32 GB
- Disk: 1 TB SSD
- OS: Windows 11

Minimum recommended to reproduce the running-example results in reasonable time:
- CPU: at least 1 core (2 cores recommended)
- RAM: at least 1 GB of RAM (4 GB recommended)
- Disk: ~2 GB free (python + packages + datasheets + figures)

### Software Requirements

- **Python** >= 3.9 (tested with version: 3.13.7)
  - **Python packages** (pip install)
    - pandas
    - numpy
    - matplotlib
    - scipy
    - statsmodels
    - pillow
- Host OS: any OS supported by Python (Linux, macOS, Windows).

```
pip install pandas numpy matplotlib scipy statsmodels pillow
```

## Project Layout

- `datasheets/` - input Excel data files.
- `scripts/` - the figure-generation scripts, plus `plot_common.py`, a shared helper module
  (statistics and figure-saving code used by all three scripts). It is not meant to be run directly.
- `results/` - output folder; created automatically the first time a script is run.

Each script reads its data files from `../datasheets` (relative to the script) and writes its
figures to `../results`, so always run the scripts from inside the `scripts` folder.

Every script accepts, as optional command-line arguments, one or more of the figure keys listed
below to (re)generate only those figures; with no arguments it generates every figure it defines.
Each figure is saved in three formats: `.tif` (300 dpi, RGB, LZW), `.pdf` (vector) and `.svg`
(vector, editable text).

## Reproducing the Figures

### Figure 2(f), Figure 2(h) and their Supplementary versions

Script: `generate_figures_2F_2H_and_supplementaries.py`
Data files: `Figure_2F_MIA.xlsx`, `Figure_2H_PANC.xlsx`, `Supl__Figure_2F.xlsx`, `Suppl__Figure_2H.xlsx`

```
cd scripts
python generate_figures_2F_2H_and_supplementaries.py
```

| Figure key        | Paper figure                |
|--------------------|------------------------------|
| `MIAPaCa-2_48h`    | Figure 2(f)                  |
| `PANC-1_48h`       | Figure 2(h)                  |
| `MIAPaCa-2_24h`    | Supplementary Figure 2(f)    |
| `PANC-1_24h`       | Supplementary Figure 2(h)    |

To generate a single figure, e.g. Figure 2(f):
```
python generate_figures_2F_2H_and_supplementaries.py MIAPaCa-2_48h
```

Output: `results/violin_plot_<figure key>.tif / .pdf / .svg`

### Figure 6(g) and Figure 6(h)

Script: `generate_figures_6G_6H.py`
Data files: `Figure_6G_MIA.xlsx`, `Figure_6H_PANC.xlsx`

```
cd scripts
python generate_figures_6G_6H.py
```

| Figure key          | Paper figure |
|----------------------|--------------|
| `MIAPaCa-2_RUVBL2`   | Figure 6(g)  |
| `PANC-1_RUVBL2`      | Figure 6(h)  |

Output: `results/split_violin_<figure key>.tif / .pdf / .svg`

### Figure 7(b)

Script: `generate_figure_7B.py`
Data file: `Figure_7B_RUVB2_WT_MUT.xlsx`

```
cd scripts
python generate_figure_7B.py
```

Output: `results/split_violin_HEK293T_RUVBL2_WT_MUT.tif / .pdf / .svg`