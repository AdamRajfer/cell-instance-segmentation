from typing import Dict, Iterable, Tuple

import numpy as np


class Encoder:
    def __init__(self) -> None:
        self._label_to_id: Dict[str, int] = {}
        self._id_to_label: Dict[int, str] = {}

    def __len__(self) -> int:
        return len(self._label_to_id)

    def fit(self, labels: Iterable[str]) -> "Encoder":
        self._label_to_id = {label: i for i, label in enumerate(sorted(set(labels)))}
        self._id_to_label = {i: label for label, i in self._label_to_id.items()}
        return self

    def encode_label(self, x: str) -> int:
        return self._label_to_id[x]

    def decode_label(self, x: int) -> str:
        return self._id_to_label[x]

    def encode_rle(self, mask: np.ndarray) -> str:
        pixels = np.concatenate([[0], mask.ravel(), [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(map(str, runs))

    def decode_rle(self, encoded_mask: str, shape: Tuple[int, int]) -> np.ndarray:
        encoded_mask = np.fromiter(encoded_mask.split(), dtype=int).reshape(-1, 2)
        encoded_mask[:, 1] += encoded_mask[:, 0]
        encoded_mask -= 1
        mask = np.zeros(np.prod(shape), dtype=int)
        for start, end in encoded_mask:
            mask[start:end] = 1
        return mask.reshape(shape)
