<div align="center">

# GPTCast: a weather language model for precipitation nowcasting

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-GMD-B31B1B.svg)](https://doi.org/10.5194/gmd-18-5351-2025)
[![Data](http://img.shields.io/badge/Data-Zenodo-4b44ce.svg)](https://doi.org/10.5281/zenodo.13692016)
[![Models](http://img.shields.io/badge/Models-Zenodo-4b44ce.svg)](https://doi.org/10.5281/zenodo.13594332)

</div>

<br>

## Description

Code release for the paper <b>"GPTCast: a weather language model for precipitation nowcasting"</b>

```
@Article{gmd-18-5351-2025,
AUTHOR = {Franch, G. and Tomasi, E. and Wanjari, R. and Poli, V. and Cardinali, C. and Alberoni, P. P. and Cristoforetti, M.},
TITLE = {GPTCast: a weather language model for precipitation nowcasting},
JOURNAL = {Geoscientific Model Development},
VOLUME = {18},
YEAR = {2025},
NUMBER = {16},
PAGES = {5351--5371},
URL = {https://gmd.copernicus.org/articles/18/5351/2025/},
DOI = {10.5194/gmd-18-5351-2025}
}
```

<b>paper</b>: [https://gmd.copernicus.org/articles/18/5351/2025/](https://doi.org/10.5194/gmd-18-5351-2025)

<b>data</b>: https://doi.org/10.5281/zenodo.13692016

<b>models</b>: https://doi.org/10.5281/zenodo.13594332


## How to run

Install dependencies

```bash
# install python3.12 on ubuntu
bash install_python_ubuntu.sh

# create environment with poetry
bash create_environment.sh

# activate the environment
source .venv/bin/activate 
```

## Use the pretrained models

Check the notebooks in the [notebooks](notebooks/) folder on how to use the pretrained models.

- See the notebook [notebooks/example_gptcast_forecast.ipynb](notebooks/example_gptcast_forecast.ipynb) for running the models on a test batch and generating a forecast.

- See the notebook [notebooks/example_autoencoder_reconstruction.ipynb](notebooks/example_autoencoder_reconstruction.ipynb) for a test on the VAE reconstruction.

## Training

To train the model on the original dataset, first run the script in the [data](data/) folder to download the dataset.

```bash
# download the dataset
python data/download_data.py
```

### Train the VAE
Train the first stage (the VAE) with one of the following configurations contained in the folder [configs/experiment/](configs/experiment/):
- [vaeganvq_mae](configs/experiment/vaeganvq_mae.yaml) - Mean Absolute Error loss
- [vaeganvq_mwae](configs/experiment/vaeganvq_mwae.yaml) - Magnitude Weighted Absolute Error loss

```bash
# train a VAE with WMAE reconstruction loss on GPU
# the result (including model checkpoints) will be saved in the folder `logs/train/`
python gptcast/train.py trainer=gpu experiment=vaeganvq_mwae.yaml 
```

### Train GPTCast
After training the VAE, train the GPTCast model with one of the following configurations contained in the folder [configs/experiment/](configs/experiment/):
- [gptcast_8x8](configs/experiment/gptcast_8x8.yaml) - 8x8 token spatial context (128x128 pixels)
- [gptcast_16x16](configs/experiment/gptcast_16x16.yaml) - 16x16 token spatial context (256x256 pixels)

```bash
# train GPTCast with a 16x16 token spatial context on GPU
# the result (including model checkpoints) will be saved in the folder `logs/train/`
# the VAE checkpoint path should be provided
python gptcast/train.py trainer=gpu experiment=gptcast_16x16.yaml model.first_stage.ckpt_path=<path_to_vae_checkpoint>
```
