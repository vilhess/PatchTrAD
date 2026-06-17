# TSB-AD — PatchTrAD

Benchmarking [PatchTrAD](PatchTrAD.py) on the [TSB-AD](https://github.com/thedatumorg/TSB-AD) anomaly detection benchmark.

This is a lightly adapted fork of the original **TSB-AD** code base
(https://github.com/thedatumorg/TSB-AD). We slightly reorganized the overall
architecture of the repository and provide an implementation only for the
**PatchTrAD** detector.

## Status

- ✅ **Univariate** time series — supported.
- 🚧 **Multivariate** time series — still in deployment, not available yet.
- Detector: **PatchTrAD** only.

## Setup

Follow the installation and dataset instructions from the original
[TSB-AD repository](https://github.com/thedatumorg/TSB-AD). Once the datasets are
in place, **set the paths correctly** in [benchmark_exp/configs.py](benchmark_exp/configs.py):

```python
DEFAULT_PATHS = {
    'dataset_dir':       '...',  # folder containing the TSB-AD-U series
    'file_list':         '...',  # evaluation file list
    'save_dir':          '...',  # where evaluation metrics are written
    'tuning_file_list':  '...',  # hyperparameter tuning file list
    'tuning_save_dir':   '...',  # where tuning results are written
}
```

## Usage

Run the commands from the `tsbad` directory:

```bash
cd tsbad

# Hyperparameter tuning
python -m benchmark_exp.HP_Tuning_U

# Evaluation with the optimal hyperparameters
python -m benchmark_exp.Run_Detector_U
```

Each path can also be overridden on the command line, e.g.
`python -m benchmark_exp.Run_Detector_U --dataset_dir /path/to/data`.

## Results

Results can be explored in [benchmark_exp/results.ipynb](benchmark_exp/results.ipynb).
Considering **VUS-ROC**, PatchTrAD ranks **second**, behind the now ICLR 2026
state-of-the-art [PaAno](https://arxiv.org/abs/2602.01359).

## Credits

Original benchmark: [thedatumorg/TSB-AD](https://github.com/thedatumorg/TSB-AD).
