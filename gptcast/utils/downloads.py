import httpx
import hashlib
from pathlib import Path
from typing import Optional
from tqdm import trange

PRETRAINED_MODELS = {
    "gptcast_16": {
        "url": "https://zenodo.org/records/13594332/files/gptcast_16.ckpt",
        "byte_size": 1473469865,
        "md5sum": "5982ee6d92c36edc3338abf566688d99",
    },
    "gptcast_8": {
        "url": "https://zenodo.org/records/13594332/files/gptcast_8.ckpt",
        "byte_size": 1469032619,
        "md5sum": "f1df51ec1157e4e6640333e01c065bea",
    },
    "vae_mae" : {
        "url": "https://zenodo.org/records/13594332/files/vae_mae.ckpt",
        "byte_size": 336873650,
        "md5sum": "0ad24472dcc76f11936cc81390486779",
    },
    "vae_mwae" : {
        "url": "https://zenodo.org/records/13594332/files/vae_mwae.ckpt",
        "byte_size": 336720590,
        "md5sum": "908d0222f37c9ac1964e72072966d957",
    },
}

DATASETS = {
    "miarad" : {
        "miarad_test.csv": {
            "url": "https://zenodo.org/record/13692016/files/miarad_test.csv",
            "byte_size": 812446,
            "md5sum": "5f293592e2760064b652ef7414274eb1",
        },
        "miarad_test.tar": {
            "url": "https://zenodo.org/record/13692016/files/miarad_test.tar",
            "byte_size": 726865920,
            "md5sum": "f5fa81ae993d7568a99adf45b6766cb7",
        },
        "miarad_training.csv": {
            "url": "https://zenodo.org/record/13692016/files/miarad_training.csv",
            "byte_size": 4455802,
            "md5sum": "b64551045c1fc4f5f65afec5fa2bdb85",
        },
        "miarad_training.tar": {
            "url": "https://zenodo.org/record/13692016/files/miarad_training.tar",
            "byte_size": 4894545920,
            "md5sum": "519b02e410620959c02530ffe60bdf46",
        },
    },
    "fts" : {
        "fts.tar": {
            "url": "https://zenodo.org/records/13692016/files/fts.tar",
            "byte_size": 43332034560,
            "md5sum": "b25b3cbe9b1d5e0f9c077af526148f94",
        },
    }
}

def download_file(url: str, path: Path, overwrite: bool = False, md5sum: Optional[str] = None, file_size: int = 0) -> Optional[str]:
    """
    Download a file from a given URL and save it to a given path.
    If file already exists on the path, it will not be downloaded again unless overwrite is True.

    Args:
        url (str): URL of the file to download.
        path (Path): Path to save the downloaded file.
        overwrite (bool): Whether to overwrite the file if it already exists.
        md5sum (str): Expected md5sum of the file.
    
    Returns:
        str: Path to the downloaded file.
    """
    if path.exists() and not overwrite:
        print(f"File {path.name} already exists on the path. Skipping download.")
        return str(path)
    if md5sum is not None:
        md5 = hashlib.md5()
    with httpx.stream("GET", url, follow_redirects=True) as response:
        with open(path, "wb") as f:
            file_size = int(response.headers["Content-Length"])  if "Content-Length" in response.headers and file_size == 0 else file_size
            progress = trange(file_size, desc=f"Downloading {path.name}", unit="B", unit_scale=True)
            for chunk in response.iter_bytes():
                f.write(chunk)
                if md5sum is not None:
                    md5.update(chunk)
                progress.update(len(chunk))
    if md5sum is not None and md5sum != md5.hexdigest():
        print(f"Failed to download {path.name} as md5sum {md5sum} does not match the downloaded file.")
        # path.unlink()
        return None
    return str(path)

def download_dataset(dataset_name: str, path: Path, overwrite: bool = False) -> bool:
    """
    Download a dataset by name.

    Args:
        dataset_name (str): Name of the dataset.
        path (Path): Path to save the downloaded dataset.
        overwrite (bool): Whether to overwrite the dataset if it already exists.

    Returns:
        bool: True if the dataset was downloaded successfully, False otherwise.
    """
    print(f"Downloading {dataset_name} dataset to {path}, overwrite={overwrite}")
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found in datasets.")
    for file_name, file_info in DATASETS[dataset_name].items():
        url = file_info["url"]
        file_path = path / file_name
        md5sum = file_info["md5sum"]
        file_size = file_info["byte_size"]
        if download_file(url, file_path, overwrite, md5sum, file_size) is None:
            return False
    print("DONE")
    return True

def download_pretrained_model(model_name: str, path: Path, overwrite: bool = False) -> Optional[str]:
    """
    Download a pretrained model by name.

    Args:
        model_name (str): Name of the pretrained model.
        path (Path): Path to save the downloaded model.
        overwrite (bool): Whether to overwrite the model if it already exists.

    Returns:
        bool: True if the model was downloaded successfully, False otherwise.
    """
    if model_name not in PRETRAINED_MODELS:
        raise ValueError(f"Model {model_name} not found in pretrained models.")
    model_info = PRETRAINED_MODELS[model_name]
    url = model_info["url"]
    file_path = path / f"{model_name}.ckpt"
    md5sum = model_info["md5sum"]
    file_size = model_info["byte_size"]
    return download_file(url, file_path, overwrite, md5sum, file_size)

def download_all_datasets(path: Path, overwrite: bool = False) -> bool:
    """
    Download all datasets to a given path.

    Args:
        path (Path): Path to save the downloaded datasets.
        overwrite (bool): Whether to overwrite the datasets if they already exist.

    Returns:
        bool: True if all datasets were downloaded successfully, False otherwise.
    """
    print(f"Downloading all datasets to {path}, overwrite={overwrite}")
    for dataset_name in DATASETS:
        if not download_dataset(dataset_name, path, overwrite):
            return False
    return True

def download_all_pretrained_models(path: Path, overwrite: bool = False) -> bool:
    """
    Download all pretrained models to a given path.

    Args:
        path (Path): Path to save the downloaded models.
        overwrite (bool): Whether to overwrite the models if they already exist.

    Returns:
        bool: True if all models were downloaded successfully, False otherwise.
    """
    print(f"Downloading all pretrained models to {path}, overwrite={overwrite}")
    for model_name in PRETRAINED_MODELS:
        if download_pretrained_model(model_name, path, overwrite) is None:
            return False
    return True