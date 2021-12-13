import os
from functools import partial
from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import Compose
from torch.utils import data

from sartorius.encoder import Encoder


class Dataset(data.Dataset):
    def __init__(
        self,
        config_df: pd.DataFrame,
        encoder: Encoder,
        transformations: Compose,
        return_bboxes: bool,
        max_bboxes: int = 1000,
    ) -> None:
        self._config_df = config_df
        self._encoder = encoder
        self._transformations = transformations
        self._return_bboxes = return_bboxes
        self._max_bboxes = max_bboxes

    def __getitem__(self, i: Union[int, str]) -> Dict[str, Any]:
        item = self._config_df.iloc[i] if isinstance(i, int) else self._config_df.loc[i]
        image = cv2.imread(item["path"])
        masks = list(map(partial(self._encoder.decode_rle, shape=image.shape[:2]), item["annotations"]))
        inputs = {"image": image, "mask": np.max(masks, axis=0), "image_id": item.name}
        if self._return_bboxes:
            inputs["bboxes"] = np.array(list(map(self._build_bbox, masks)))
        inputs = self._transformations(**inputs)
        inputs["mask"] = inputs["mask"].float()
        if self._return_bboxes:
            inputs["bboxes"] = np.concatenate(
                [inputs["bboxes"], np.zeros((max(self._max_bboxes - len(inputs["bboxes"]), 0), 4))]
            )
        inputs["label"] = torch.tensor(self._encoder.encode_label(item["cell_type"]))
        return inputs

    def __len__(self) -> int:
        return len(self._config_df)

    def _build_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        y_indexes, x_indexes = np.where(mask == 1)
        return np.min(x_indexes), np.min(y_indexes), np.max(x_indexes), np.max(y_indexes)


class SubmissionDataset(data.Dataset):
    def __init__(self, data_dir: str, transformations: Compose) -> None:
        paths = sorted(os.listdir(data_dir))
        self._image_paths = [os.path.join(data_dir, path) for path in paths]
        self._image_ids = [path.split(".")[0] for path in paths]
        self._transformations = transformations

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return {
            "image": self._transformations(image=cv2.imread(self._image_paths[i]))["image"],
            "image_id": self._image_ids[i],
        }

    def __len__(self) -> int:
        return len(self._image_paths)
