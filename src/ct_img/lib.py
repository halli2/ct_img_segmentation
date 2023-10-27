from pathlib import Path
from typing import Optional
from pydicom.dataset import FileDataset
import pydicom
import numpy as np
from PIL import Image
from dataclasses import dataclass
from numpy import typing as npt


@dataclass
class Roi:
    y0: int
    y1: int
    x0: int
    x1: int
    z0: int
    z1: int


def normalize_image(
    image: float | int,
    window_max: int,
    window_min: int,
    max_value: int = 255,
) -> int:
    """Normalize an image between 0 and *max_value*, usually 255 for 8bit"""
    if image > window_max:
        return max_value
    elif image < window_min:
        return 0
    else:
        return int(
            ((image - window_min) / (window_max - window_min)) * max_value,
        )


def convert(
    subject: str,
    # Window center, width
    custom_window: Optional[tuple[int, int]] = None,
) -> None:
    """Converts .dcm images to uint8 png images."""
    subject_path = Path(f"image_data/{subject}")
    saved_folder = Path(f"processed_images/{subject}")
    saved_folder.mkdir(parents=True, exist_ok=True)
    file_paths = sorted(subject_path.glob("*dcm"))

    normalize_func = np.vectorize(normalize_image)
    for i, file_name in enumerate(file_paths):
        with pydicom.dcmread(file_name) as ds:
            # if not hasattr(ds, "SliceLocation"):
            #     print(f"Skipped file {file_name}: No sliceLocation")
            #     continue
            if custom_window is not None:
                center = custom_window[0]
                width = custom_window[1]
            elif type(ds.WindowCenter) is list:
                center, width = int(ds.WindowCenter[0]), int(ds.WindowWidth[0])
            else:
                center, width = int(ds.WindowCenter), int(ds.WindowWidth)
            rescale_slope = int(ds.RescaleSlope)
            rescale_intercept = int(ds.RescaleIntercept)

            window_max = int(center + width / 2)
            window_min = int(center - width / 2)
            pixels = ds.pixel_array * rescale_slope + rescale_intercept

            pixels = normalize_func(pixels, window_max, window_min)
            pixels = np.array(pixels, dtype=np.uint8)
            image = Image.fromarray(pixels)

            saved_path = saved_folder.joinpath(f"{i}.png")
            image.save(saved_path)
            if (i + 1) % 10 == 0:
                print(f"Converted {i + 1}")


def create_3d_image(roi: Roi, patient: str):
    img_shape = [roi.y1 - roi.y0, roi.x1 - roi.x0, roi.z1 - roi.z0]
    image_3d = np.zeros(img_shape, dtype=np.uint8)
    for i in range(roi.z1 - roi.z0):
        with Image.open(f"processed_images/{patient}/{i}.png") as im:
            image_3d[:, :, i] = np.asarray(im)[roi.y0 : roi.y1, roi.x0 : roi.x1]
    return image_3d


def sag_view(img: npt.NDArray, x: int) -> npt.NDArray:
    """Get a sagittal view in 3d image"""
    shape = img.shape
    tmp = img[:, x, :]
    sag = np.rot90(np.reshape(tmp, (shape[0], shape[2])))
    return sag


def cor_view(img: npt.NDArray, y: int) -> npt.NDArray:
    """Get a coronal view in the 3d image"""
    shape = img.shape
    tmp = img[y, :, :]
    cor = np.rot90(np.reshape(tmp, (shape[1], shape[2])))
    return cor
