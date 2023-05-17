from dataclasses import dataclass
from typing import Type

import numpy as np
from numpy.typing import NDArray


@dataclass
class MotionDataSample:
    t: float
    q: np.ndarray
    fields = {}

    def __getattr__(self, __name: str) -> NDArray:
        range = self.fields[__name]
        return self.q[range[0] : range[1]]


@dataclass
class KeyframeMotionDataSample:
    t: float
    q: np.ndarray
    qdot: np.ndarray
    dt: float

    fields = {}
    BaseSampleType: Type[MotionDataSample] = MotionDataSample

    def __getattr__(self, __name: str) -> NDArray:
        range = self.fields[__name]
        return self.q[range[0] : range[1]]


class IterableKeyframeMotionDataset:
    SampleType: Type[KeyframeMotionDataSample] = KeyframeMotionDataSample

    def __init__(self) -> None:
        pass

    def __iter__(self):
        raise NotImplementedError


class MapKeyframeMotionDataset(IterableKeyframeMotionDataset):
    SampleType = KeyframeMotionDataSample

    def __init__(self) -> None:
        pass

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> SampleType:
        raise NotImplementedError

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def duration(self) -> float:
        return self[-1].t + self[-1].dt - self[0].t
