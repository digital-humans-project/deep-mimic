from dataclasses import dataclass
from typing import Dict

import numpy as np
from numpy.typing import NDArray


@dataclass
class MotionDataSample:
    t: float
    q: np.ndarray
    fields: Dict = {}


@dataclass
class KeyframeMotionDataSample:
    t: float
    q: np.ndarray
    dt: float
    fields: Dict = {}
    
    def __getattr__(self, __name: str) -> NDArray:
        range = self.fields[__name]
        return self.q[range[0]:range[1]]
