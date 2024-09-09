from pysteps.visualization import plot_precip_field as plot_field
import matplotlib.pyplot as plt
import numpy as np
from typing import Union

def plot_miarad(arr: Union[np.ndarray, np.ma.MaskedArray], colorbar=False, colorscale='STEPS-BE', use_utm_projection: bool = True, dpi: int = 200, title: str = '', figsize=None):
    assert arr.ndim in [2, 3], "Array must be 2D or 3D (time, height, width)"
    plt.rcParams.update({'font.size': 6, 'font.weight': 'normal'})

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
        steps = 1
    else:
        steps = arr.shape[0]

    # add 20% width if colorbar is True
    x_size = 1.2 if colorbar else 1
    if use_utm_projection:
        #utm 32632
        geodata = {
            "projection": "+proj=utm +zone=32 +datum=WGS84 +units=m +no_defs +type=crs",
            "x1": 458995.938,
            "y1": 4804873.924,
            "x2": 841529.075,
            "y2": 5103289.827,
            "yorigin":"lower"
        }
        # bbox in (lower left x, lower left y ,upper right x, upper right y)
        bbox = (geodata['x1']+32000, geodata['y1']+15000, geodata['x2'], geodata['y2']-10000)
        figsize = ((bbox[2]-bbox[0])/100000*x_size, (bbox[3]-bbox[1])/100000) if figsize is None else figsize
    else:
        geodata = {
            "projection": "+proj=longlat +datum=WGS84 +no_defs +type=crs",
            "x1": 8.4936750,
            "y1": 43.3955015,
            "x2": 13.2121251,
            "y2": 46.0054994,
            "yorigin":"lower"
        }
        bbox = (geodata['x1']+0.40, geodata['y1']+0.15, geodata['x2'], geodata['y2']-0.1)
        figsize=((bbox[2]-bbox[0]), bbox[3]-bbox[1]) if figsize is None else figsize

    for i in range(steps):
        titl = f"{title} +{i+1} steps" if steps > 1 else title
        figure, axis = plt.subplots(1, 1, layout="constrained", figsize=figsize, dpi=dpi)
        axis.axis('off')
        axis = plot_field(arr[i], title=titl, bbox=bbox, ax=axis, colorbar=colorbar, colorscale=colorscale, geodata=geodata,  axis='off', map_kwargs=dict(scale="10m", lw=1, drawlonlatlines=True))
        figure.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)
        plt.show()
        plt.tight_layout()


def plot_mutiple(arr: Union[np.ndarray, np.ma.MaskedArray], colorbar=False, colorscale='STEPS-BE', dpi: int = 200, title: str = '', figsize=None):
    assert arr.ndim == 3, "Array must be 3D (time, height, width)"
    figure, axis = plt.subplots(1, 1, layout="constrained", figsize=figsize, dpi=dpi)
    axis.axis('off')
    axis = plot_field(np.flipud(np.ma.concatenate(arr, axis=1)), title=title, ax=axis, colorbar=colorbar, colorscale=colorscale, axis='off')
    figure.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)
    plt.show()
    plt.tight_layout()
