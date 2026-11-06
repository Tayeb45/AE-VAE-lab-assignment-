# Assignment 2 — reconstructing CAMUS echocardiography with a VAE

Tayeb BOUCHIKHI — IAV, INSA Lyon

Nine experiments train a MONAI `AutoencoderKL` on CAMUS 64×64. The write-up is `final_notebook.ipynb`. The runnable code is the notebooks under `experiments/`.

This archive contains no dataset, no trained weights, and no image files. Plots are drawn again when a notebook is executed. The numbers in the report come from those runs.

## Setup

Tested with Python 3.12. A CUDA GPU is what experiments 4 to 9 were trained on (RTX 4060 Ti). They also run on CPU, much more slowly. Experiment 1 to 3 are the CPU runs.

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
pip install torch torchvision
pip install -r requirements.txt
```

Linux or macOS:

```bash
source .venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
```

If you have a GPU, install the matching CUDA wheel from [pytorch.org](https://pytorch.org) instead of the default CPU build. Each notebook uses `cuda` when it is available.

## Data

Images stay outside this archive. Point `CAMUS_DATA_ROOT` at the directory that **contains** the `camus_64` folder:

```powershell
$env:CAMUS_DATA_ROOT = "C:\path\to\parent_of_camus_64"
```

```bash
export CAMUS_DATA_ROOT=/path/to/parent_of_camus_64
```

If that folder is missing, the notebook downloads `camus_64.zip` from the course server into `./data/camus_64`. Do not put the dataset in the submitted archive.

## Run

Run the notebooks from **this directory** (the folder that contains `experiments/`). Checkpoints and logs are written to `checkpoints/` and `artifacts/`, relative to the current working directory. Those folders are created locally and are not part of the submission.

Open the notebook and run all cells, or:

```bash
jupyter nbconvert --to notebook --execute --inplace experiments/cpu/exp_1_vae_camus.ipynb --ExecutePreprocessor.timeout=14400
```

Same command for the other notebooks. Run them in order. Experiment 9 reads `artifacts/results_exp4.json` and `artifacts/results_exp5.json` for its comparison, so experiments 4 and 5 have to be finished first. The copies of those JSON files shipped here are the text logs of the finished runs, so that comparison cell can run even before a new training.

| Notebook | Device used | What it trains | Skip training |
| --- | --- | --- | --- |
| `experiments/cpu/exp_1_vae_camus.ipynb` | CPU | β ablation, VAE, AE, VQ-VAE | set `TRAIN = False` (needs `checkpoints/`) |
| `experiments/cpu/exp_2_vae_camus.ipynb` | CPU | wider VAE, MSE + KL | `CAMUS_EXP2_TRAIN=0` |
| `experiments/cpu/exp_3_vae_camus.ipynb` | CPU | ranking loss, γ = 10⁻³ | `CAMUS_EXP3_TRAIN=0` |
| `experiments/gpu/exp_4_vae_camus.ipynb` | GPU | weighted loss, γ = 0.05 | `CAMUS_EXP4_TRAIN=0` |
| `experiments/gpu/exp_5_vae_camus.ipynb` | GPU | reference model | `CAMUS_EXP5_TRAIN=0` |
| `experiments/gpu/exp_6_vae_camus.ipynb` | GPU | 49 M parameters | `CAMUS_EXP6_TRAIN=0` |
| `experiments/gpu/exp_7_vae_camus.ipynb` | GPU | three latent axes | `CAMUS_EXP7_TRAIN=0` |
| `experiments/gpu/exp_8_vae_camus.ipynb` | GPU | bilinear augmentation, frozen encoder | `CAMUS_EXP8_TRAIN=0` |
| `experiments/gpu/exp_9_vae_camus.ipynb` | GPU | model kept in the report | `CAMUS_EXP9_TRAIN=0` |

`CAMUS_EXPn_TRAIN=0` only works if `checkpoints/expN_vae_best.pth` is already on disk. Those weight files are binary and are **not** in this archive. Leave training on for a clean run.

Experiment 1 epoch counts, without editing the notebook:

```bash
export CAMUS_ABLATION_EPOCHS=8
export CAMUS_FINAL_EPOCHS=25
export CAMUS_AE_EPOCHS=20
export CAMUS_VQ_EPOCHS=20
```

On a laptop CPU, experiment 1 takes about 100 minutes. Experiments 4 to 9 take a few minutes each on the GPU used here, and much longer on CPU.

## Trained weights

Not included, per the submission rules. The checkpoint kept for the report is epoch 109 of experiment 9.
