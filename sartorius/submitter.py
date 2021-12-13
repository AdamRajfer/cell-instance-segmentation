from itertools import chain
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from sartorius.encoder import Encoder


class Submitter:
    def __init__(self, encoder: Encoder) -> None:
        self._encoder = encoder

    def prepare_for_submission(self, outputs: Dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame(
            chain(*(self.encode(data["mask"], image_id) for image_id, data in outputs.items())),
            columns=["id", "predicted"],
        )

    def encode(self, prediction: np.ndarray, image_id: str) -> List[Tuple[str, str]]:
        num_components, components = cv2.connectedComponents((prediction > 0.5).astype(np.uint8))
        if num_components < 2:
            return [(image_id, "")]
        return [(image_id, self._encoder.encode_rle((components == i).astype(int))) for i in range(1, num_components)]
