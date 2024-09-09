import numpy as np
from datetime import datetime, timedelta
import os
import gzip
import pandas as pd
from typing import Union, Tuple
from torch.utils.data import Dataset
import tarfile
import netCDF4 as nc
from tqdm import tqdm
import logging
import ast
import einops
import albumentations as A


def normalized_reflectivity_to_rainrate(arr: np.ndarray,
                                        minmax: tuple = (-20, 60),
                                        a: float = 200.0,
                                        b: float = 1.6):
    """
    Input is 0 - 1 normalized reflectivity value
        ( reflectivity (dbZ) / max reflectivity (52.5) )
    Output is mm/h rain rate
    """
    min, max = minmax
    rescaled = (arr * (max - min)) + min
    return reflectivity_to_rainrate(rescaled, a, b)


def reflectivity_to_rainrate(arr: np.ndarray, a: float = 200.0, b: float = 1.6):
    Z = 10.0 ** (arr / 10.0)  # wradlib.trafo.idecibel
    return (Z / a) ** (1.0 / b)  # wradlib.zr.z_to_r


class MiaradDataset(Dataset):
    date_fmt = 'COMP_%Y%m%d%H%M'
    gap = timedelta(minutes=5)

    @staticmethod
    def generate_metadata(tarfile_path: str) -> pd.DataFrame:
        files = []
        tar = tarfile.open(tarfile_path)
        for f in tar:
            if f.name.endswith('.nc.gz'):
                files.append(os.path.basename(f.name[:-6]))
        files.sort()
        dirname = os.path.dirname(f.name)

        sequences = []
        curr_seq = None
        for f in tqdm(files, desc='metadata extraction', leave=True):

            ds = nc.Dataset(f, memory=gzip.decompress(tar.extractfile(f'{dirname}/{f}.nc.gz').read()), mode='r')[
                     'Z_60'][:]
            npixels = np.prod(ds.data.shape)
            data_norm = (ds.data + 20) / 80 * ~ds.mask  # data range goes from -20 to 60 dbz
            data_sum = float(data_norm[data_norm > 0.01].sum())
            dt_curr = datetime.strptime(f, MiaradDataset.date_fmt)

            if curr_seq is None:
                curr_seq = {'start_datetime': dt_curr, 'pixel_average': data_sum, 'files': [f'{dirname}/{f}']}
            elif dt_curr - MiaradDataset.gap > dt_prev:
                curr_seq['end_datetime'] = dt_prev
                curr_seq['seq_length'] = int(
                    (curr_seq['end_datetime'] - curr_seq['start_datetime']) / MiaradDataset.gap) + 1
                curr_seq['pixel_average'] /= (npixels * curr_seq['seq_length'])
                sequences.append(curr_seq)
                curr_seq = {'start_datetime': dt_curr, 'pixel_average': data_sum, 'files': [f'{dirname}/{f}']}
            else:
                curr_seq['pixel_average'] += data_sum
                curr_seq['files'].append(f'{dirname}/{f}')

            dt_prev = dt_curr
        curr_seq['end_datetime'] = dt_prev
        curr_seq['seq_length'] = int((curr_seq['end_datetime'] - curr_seq['start_datetime']) / MiaradDataset.gap) + 1
        curr_seq['pixel_average'] /= (npixels * curr_seq['seq_length'])
        sequences.append(curr_seq)

        df = pd.DataFrame(sequences)
        df.index.name = 'id'

        return df

    @staticmethod
    def parse_metadata(csv_path: str) -> pd.DataFrame:
        return pd.read_csv(csv_path, index_col='id', converters={'files': ast.literal_eval})

    def __init__(self, tarfile_path: str, metadata_path_or_df: Union[str, pd.DataFrame] = None,
                 padding: Union[int, tuple] = None):
        self.__tar = tarfile.open(tarfile_path)

        if metadata_path_or_df is None:
            logging.info("Computing metadata...")
            self.meta = self.generate_metadata(tarfile_path)
            logging.info("OK")
        elif type(metadata_path_or_df) is str:
            self.meta = self.parse_metadata(metadata_path_or_df)
            self.meta['start_datetime'] = pd.to_datetime(self.meta['start_datetime'])
            self.meta['end_datetime'] = pd.to_datetime(self.meta['end_datetime'])
        else:
            self.meta = metadata_path_or_df

        self.__set_variables()
        self.__padding = None
        self.__inner_shape = self.__read_data(self.__files[0]).data.shape

        self.__padding = padding  # (bottom_rows, right_columns)
        if self.__padding is not None:
            if type(self.__padding) is int:
                rowpad = colpad = self.__padding
            else:
                rowpad, colpad = self.__padding

            self.__pad_rows = list(self.__inner_shape)  # (1, rows, cols)
            self.__pad_rows[-2] = rowpad
            self.__pad_rows = np.ma.array(np.ones(self.__pad_rows) * -20, mask=np.ones(self.__pad_rows))

            self.__pad_cols = list(self.__inner_shape)  # (1, rows, cols)
            self.__pad_cols[-2] += rowpad
            self.__pad_cols[-1] = colpad
            self.__pad_cols = np.ma.array(np.ones(self.__pad_cols) * -20, mask=np.ones(self.__pad_cols))

            self.__inner_shape = *self.__inner_shape[:-2], self.__inner_shape[-2] + rowpad, self.__inner_shape[
                -1] + colpad

    @property
    def shape(self):
        return (self.__len, *self.__inner_shape)

    def __read_data(self, fname: str, num_retries: int = 1):
        f = f'{fname}.nc.gz'

        for attempt_no in range(num_retries):
            try:
                data = nc.Dataset(f, memory=gzip.decompress(self.__tar.extractfile(f).read()), mode='r')['Z_60'][:]
                break
            except Exception as e:
                if attempt_no < (num_retries - 1):
                    logging.debug(f"Error reading file {f} from tar {self.__tar}")
                else:
                    logging.warning(f"Error reading file {f} from tar {self.__tar} for {num_retries} times! Using zeroes!")
                    data = np.ma.ones((1, 290, 373), dtype=np.float32) * -20
                    data.mask = np.ones(data.shape).astype(bool)

        if self.__padding is not None:
            data = np.ma.concatenate((data, self.__pad_rows), axis=-2)
            data = np.ma.concatenate((data, self.__pad_cols), axis=-1)
        return data

    def __set_variables(self):
        files = []
        for f in self.meta['files']:
            files.extend(f)
        self.__files = files

        self.__timestamps = [datetime.strptime(os.path.basename(f), MiaradDataset.date_fmt) for f in files]
        self.__len = len(self.__files)

    def __len__(self) -> int:
        return self.__len

    def __getitem__(self, i: int) -> Tuple[datetime, np.ma.masked_array]:
        arr = self.__read_data(self.__files[i], 3)
        return self.__timestamps[i], arr

    def save_metadata(self, csv_path: str):
        self.meta.to_csv(csv_path, columns=['start_datetime', 'end_datetime', 'seq_length', 'pixel_average', 'files'])

    def set_metadata(self, metadata: pd.DataFrame):
        self.meta = metadata
        self.__set_variables()


class Miarad(MiaradDataset):
    def __init__(self, crop=256, clip_and_normalize=None, *args, **kwargs):
        # self.clip_and_normalize = None if 'clip_and_normalize' not in kwargs.keys() else kwargs.pop(
        #     'clip_and_normalize')
        self.clip_and_normalize = clip_and_normalize
        self.crop = crop
        super(Miarad, self).__init__(*args, **kwargs)

    def preprocess_image(self, arr: np.ma.masked_array):
        image = einops.rearrange(arr.data, 'c h w -> h w c')
        mask = arr.mask
        if self.crop is not None:
            transform = A.Compose([
                A.RandomCrop(width=self.crop, height=self.crop),
                A.RandomRotate90(),
            ], additional_targets={'mask': 'image'})
            t = transform(image=image, mask=mask)
            image = t['image']
            mask = t['mask']

        if self.clip_and_normalize is not None:
            cmin, cmax, nmin, nmax = self.clip_and_normalize
            image = np.clip(image, cmin, cmax)  # clip
            image = (image - cmin) / (cmax - cmin)  # 0-1 minmax scaling
            if nmin != 0 or nmax != 1:  # rescale if necessary
                image = image * (nmax - nmin) + nmin

        return image, mask

    def __getitem__(self, i):
        ts, arr = super().__getitem__(i)
        example = dict()
        image, mask = self.preprocess_image(arr)
        example["image"] = image
        example["mask"] = mask
        example["file_path_"] = ts.strftime(self.date_fmt)
        return example


class MiaradN(MiaradDataset):
    def __init__(self, seq_len: int, stack_seq: str = None, clip_and_normalize: tuple = None, crop: int = 256, smart_crop: bool = False, **kwargs):
        super(MiaradN, self).__init__(**kwargs)
        assert seq_len > 1
        self.seq_len = seq_len
        self.future_steps = seq_len - 1
        self.stack = stack_seq
        self.clip_and_normalize = clip_and_normalize
        self.crop = crop
        self.smart_crop = smart_crop  # only crop in areas where the mask is all valid (no missing data)

    def preprocess_image(self, image: np.ndarray, mask: np.ndarray):
        # image is H x W x C
        # mask is H x W
        image = einops.rearrange(image, 'c h w -> h w c')
        if self.crop is not None:
            try:
                if self.smart_crop:
                    # random crop 30 times and take the fist where mask is all zeros
                    # crop the center if no valid crop is found
                    for _ in range(30):
                        transform = A.Compose([
                            A.RandomCrop(width=self.crop, height=self.crop),
                            A.RandomRotate90(),
                        ], additional_targets={'mask': 'image'})
                        t = transform(image=image, mask=mask)
                        if t['mask'].sum() == 0:
                            break
                    else:
                        transform = A.Compose([
                            A.CenterCrop(width=self.crop, height=self.crop),
                            A.RandomRotate90(),
                        ], additional_targets={'mask': 'image'})
                        t = transform(image=image, mask=mask)
                else:
                    transform = A.Compose([
                        A.RandomCrop(width=self.crop, height=self.crop),
                        A.RandomRotate90(),
                    ], additional_targets={'mask': 'image'})
                    t = transform(image=image, mask=mask)
            except ValueError:
                print(image.shape, mask.shape)
                raise
            image = t['image']
            mask = t['mask']

        if self.clip_and_normalize is not None:
            cmin, cmax, nmin, nmax = self.clip_and_normalize
            image = np.clip(image, cmin, cmax)  # clip
            image = (image - cmin) / (cmax - cmin)  # 0-1 minmax scaling
            if nmin != 0 or nmax != 1:  # rescale if necessary
                image = image * (nmax - nmin) + nmin

        return image, mask

    def __getitem__(self, i):
        ts, arr = super().__getitem__(i)
        samples = [arr.data]
        mask = arr.mask

        if i <= len(self) - self.seq_len:
            td = timedelta(minutes=5)
            for j in range(1, self.seq_len):
                ts_step, sample_step = super().__getitem__(i+j)
                if ts + self.gap*j != ts_step:
                    break
                else:
                    mask = np.any(mask + sample_step.mask, axis=0)
                    samples.append(sample_step.data)

        if len(samples) < self.seq_len:
            samples = np.ones_like(samples[0]).repeat(self.seq_len, axis=0) * samples[0].min()
        else:
            samples = np.concatenate(samples)*(~mask).astype(float)

        example = dict()
        image, mask = self.preprocess_image(samples, mask.squeeze())
        example["image"] = image
        example["mask"] = mask
        example["file_path_"] = ts.strftime(self.date_fmt)
        if self.stack == 'v':
            example["image"] = np.concatenate(einops.rearrange(example["image"], 'h w c -> c h w'), axis=0)[:,:,np.newaxis]
        elif self.stack == 'h':
            example["image"] = np.concatenate(einops.rearrange(example["image"], 'h w c -> c h w'), axis=1)[:,:,np.newaxis]

        return example
