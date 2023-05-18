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


class Fields:
    def __init__(self, data: NDArray) -> None:
        self.data = data

    FieldNames: ClassVar[Type[Enum]] = StrEnum
    fields: ClassVar[Dict[FieldNames, Tuple[int, int]]] = {}

    def __getattr__(self, __name: str) -> NDArray:
        return self[__name]

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in self.fields:
            self[__name] = __value
        else:
            super().__setattr__(__name, __value)

    def __getitem__(self, __name: Union[FieldNames, str]) -> NDArray:
        range = self.fields[self.FieldNames(__name)]
        return self.data[range[0] : range[1]]

    def __setitem__(self, __name: Union[FieldNames, str], value: NDArray) -> None:
        range = self.fields[self.FieldNames(__name)]
        self.data[range[0] : range[1]] = value


@dataclass
class MotionDataSample:
    t: float
    q: np.ndarray
    qdot: np.ndarray
    phase: float = 0

    FieldsType: ClassVar[Type[Fields]] = Fields

    @property
    def q_fields(self) -> FieldsType:
        return self.FieldsType(self.q)

    @property
    def qdot_fields(self) -> FieldsType:
        return self.FieldsType(self.qdot)


@dataclass
class KeyframeMotionDataSample:
    t0: float
    q0: np.ndarray
    q1: np.ndarray
    qdot: np.ndarray
    dt: float
    phase0: float = 0
    phase1: float = 0

    FieldsType: ClassVar[Type[Fields]] = Fields
    BaseSampleType: ClassVar[Type[MotionDataSample]] = MotionDataSample

    @property
    def q0_fields(self) -> FieldsType:
        return self.FieldsType(self.q0)

    @property
    def q1_fields(self) -> FieldsType:
        return self.FieldsType(self.q1)

    @property
    def qdot_fields(self) -> FieldsType:
        return self.FieldsType(self.qdot)

    @property
    def t1(self) -> float:
        return self.t0 + self.dt


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
        return self[-1].t1 - self[0].t0
