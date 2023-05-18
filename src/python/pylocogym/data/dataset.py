from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, ClassVar, Dict, Iterator, Tuple, Type, Union

import numpy as np
from numpy.typing import NDArray


class StrEnum(str, Enum):
    """
    Enum class for motion data field names.
    """

    def __new__(cls, value, *args, **kwargs):
        if not isinstance(value, (str, auto)):
            raise TypeError(f"Values of StrEnums must be strings: {value!r} is a {type(value)}")
        return super().__new__(cls, value, *args, **kwargs)

    def __str__(self) -> str:
        return str(self.value)

    def _generate_next_value_(name, *_):
        return name.lower()


@dataclass
class MotionDataSample:
    t: float
    q: np.ndarray
    qdot: np.ndarray

    Fields: ClassVar[Type[Enum]] = StrEnum
    fields: ClassVar[Dict[Fields, Tuple[int, int]]] = {}

    def __getattr__(self, __name: str) -> NDArray:
        return self[__name]

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in self.fields:
            self[__name] = __value
        else:
            super().__setattr__(__name, __value)

    def __getitem__(self, __name: Union[Fields, str]) -> NDArray:
        range = self.fields[self.Fields(__name)]
        return self.q[range[0] : range[1]]

    def __setitem__(self, __name: Union[Fields, str], value: NDArray) -> None:
        range = self.fields[self.Fields(__name)]
        self.q[range[0] : range[1]] = value


@dataclass
class KeyframeMotionDataSample:
    t: float
    q: np.ndarray
    qdot: np.ndarray
    dt: float

    Fields: ClassVar[Type[Enum]] = StrEnum
    fields: ClassVar[Dict[Fields, Tuple[int, int]]] = {}

    BaseSampleType: ClassVar[Type[MotionDataSample]] = MotionDataSample

    def __getattr__(self, __name: str) -> NDArray:
        return self[__name]

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in self.fields:
            self[__name] = __value
        else:
            super().__setattr__(__name, __value)

    def __getitem__(self, __name: Union[Fields, str]) -> NDArray:
        range = self.fields[self.Fields(__name)]
        return self.q[range[0] : range[1]]

    def __setitem__(self, __name: Union[Fields, str], value: NDArray) -> None:
        range = self.fields[self.Fields(__name)]
        self.q[range[0] : range[1]] = value


class IterableKeyframeMotionDataset:
    SampleType: Type[KeyframeMotionDataSample] = KeyframeMotionDataSample

    def __init__(self) -> None:
        pass

    def __iter__(self) -> Iterator[SampleType]:
        raise NotImplementedError


class MapKeyframeMotionDataset(IterableKeyframeMotionDataset):
    SampleType = KeyframeMotionDataSample

    def __init__(self) -> None:
        pass

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> SampleType:
        raise NotImplementedError

    def __iter__(self) -> Iterator[SampleType]:
        for i in range(len(self)):
            yield self[i]

    @property
    def duration(self) -> float:
        return self[-1].t + self[-1].dt - self[0].t
