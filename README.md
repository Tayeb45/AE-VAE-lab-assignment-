# Assignment 2 — reconstructing CAMUS echocardiography with a VAE

Train a MONAI `AutoencoderKL` variational autoencoder on the CAMUS 64×64 dataset, compare it with a deterministic autoencoder and a VQ-VAE, and evaluate reconstruction on a patient-wise held-out test set.

The runnable code is `assignment-2-camus.ipynb`. This repository does not contain images, NIfTI files, or trained weights.

## Environment

Tested with Python 3.12, PyTorch 2.15 (CPU), torchvision 0.29, and MONAI 1.6.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Use the default PyTorch index instead of the CPU wheel if you have a GPU. The notebook selects `cuda` when it is available.

## Data

Images stay outside this project. Point `CAMUS_DATA_ROOT` at the directory that contains the `camus_64` folder (one subdirectory per patient):

```bash
export CAMUS_DATA_ROOT=/path/to/parent_of_camus_64
```

If that folder is missing, the notebook downloads `camus_64.zip` from the course server into `./data/camus_64`. Do not include the dataset in the submitted archive.

## Run

From this directory:

```bash
export CAMUS_DATA_ROOT=/path/to/parent_of_camus_64
jupyter nbconvert --to notebook --execute --inplace assignment-2-camus.ipynb \
  --ExecutePreprocessor.timeout=14400
```

Or open the notebook and run all cells. The first code cells print the library versions and the train, validation, and test sizes.

On a laptop CPU the full run takes about 90–110 minutes. It took about 100 minutes on an Intel i5-1135G7. A GPU is much faster. Epoch counts can be overridden without editing the notebook:

```bash
export CAMUS_ABLATION_EPOCHS=8
export CAMUS_FINAL_EPOCHS=25
export CAMUS_AE_EPOCHS=20
export CAMUS_VQ_EPOCHS=20
```

Set `TRAIN = False` in the training section to reload `checkpoints/*.pth` instead of training again. Those files are created locally and are not part of the submission. Upload them separately if you want to share the weights, and put the link here:

- Trained weights: not included.

Plots are embedded in the executed notebook. Animations, if written, go to `artifacts/` and are not part of the archive.

## What the notebook does

1. Patient-wise 80/10/10 split of the 500 CAMUS exams (the same patient never appears in two splits).
2. MONAI `AutoencoderKL` with a 4×8×8 latent space, trained with an L1 reconstruction term and a KL term (linear warm-up).
3. Ablation of the KL weight over `{0, 1e-4, 1e-3, 1e-2, 1e-1}`.
4. Deterministic autoencoder (same backbone, no sampling and no KL) and a MONAI `VQVAE`.
5. Test metrics: MAE, MSE, PSNR, SSIM, plus a breakdown by view, cardiac instant, image quality, and inside versus outside the expert mask.
6. Latent-space analysis: t-SNE, samples from the prior, and interpolation from end-diastole to end-systole.

## Citation

Any use of CAMUS must cite:

S. Leclerc, E. Smistad, J. Pedrosa, A. Ostvik, et al. "Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography". IEEE Transactions on Medical Imaging, vol. 38, no. 9, pp. 2198–2210, Sept. 2019. doi: 10.1109/TMI.2019.2900516
