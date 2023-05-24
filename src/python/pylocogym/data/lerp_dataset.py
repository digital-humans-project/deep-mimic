try:
    from itertools import pairwise  # type: ignore
except ImportError:
    from itertools import tee

    def pairwise(iterable):
        # pairwise('ABCDEFG') --> AB BC CD DE EF FG
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


from typing import List, Optional, Tuple

import numpy as np

from pylocogym.data.dataset import IterableKeyframeMotionDataset, KeyframeMotionDataSample, MotionDataSample


class ContinuousMotionDataset:
    """
    `ContinuousMotionDataset` is the base class for all continuous motion datasets.

    Continuous motion datasets are datasets that can be evaluated at any time `t` and
    return a `MotionDataSample` that contains the state of the motion at time `t`.
    """

    def __init__(self, kf_dataset: IterableKeyframeMotionDataset) -> None:
        self.kf_dataset = kf_dataset

    def reset(self) -> None:
        """
        Reset the dataset to the beginning.
        """
        raise NotImplementedError

    def eval(self, t: float) -> Optional[MotionDataSample]:
        """
        Evaluate the motion at time `t`.

        If `t` is outside the range of the dataset, `None` is returned.
        """
        raise NotImplementedError


def lerp(a, b, alpha):
    return a * (1 - alpha) + b * alpha


def normalize(v):
    return v / np.linalg.norm(v)


def quaternion_slerp(
    quat0,
    quat1,
    fraction,
    spin=0,
    shortest_path=True,
    eps=np.finfo(float).eps * 4.0,
):
    """Return spherical linear interpolation between two quaternions.

    >>> q0 = random_quaternion()
    >>> q1 = random_quaternion()
    >>> q = quaternion_slerp(q0, q1, 0)
    >>> numpy.allclose(q, q0)
    True
    >>> q = quaternion_slerp(q0, q1, 1, 1)
    >>> numpy.allclose(q, q1)
    True
    >>> q = quaternion_slerp(q0, q1, 0.5)
    >>> angle = math.acos(numpy.dot(q0, q))
    >>> numpy.allclose(2, math.acos(numpy.dot(q0, q1)) / angle) or \
        numpy.allclose(2, math.acos(-numpy.dot(q0, q1)) / angle)
    True

    """
    q0 = normalize(quat0[:4])
    q1 = normalize(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < eps:
        return q0
    if shortest_path and d < 0.0:
        # invert rotation
        d = -d
        q1 = -q1
    angle = np.arccos(d) + spin * np.pi
    if abs(angle) < eps:
        return q0
    isin = 1.0 / np.sin(angle)
    q0 *= np.sin((1.0 - fraction) * angle) * isin
    q1 *= np.sin(fraction * angle) * isin
    q0 += q1
    return q0


def alerp(a, b, alpha, shortest_path=True):
    """
    Angular linear interpolation between two angles `a` and `b` with weight `alpha`.
    """
    if shortest_path:
        pi2 = 2 * np.pi
        da = (b - a) % pi2
        return a + alpha * (2 * da % pi2 - da)
    else:
        return a + alpha * (b - a)


class LerpMotionDataset(ContinuousMotionDataset):
    """
    `LerpMotionDataset` is a wrapper around `IterableKeyframeMotionDataset` that
    interpolates between keyframes using linear interpolation.
    """

    def __init__(
        self,
        kf_dataset: IterableKeyframeMotionDataset,
        lerp_fields: List[str] = [],
        alerp_fields: List[str] = [],
        slerp_fields: List[str] = [],
        alerp_shortest_path: bool = True,
        slerp_spin: int = 0,
        slerp_shortest_path: bool = True,
    ) -> None:
        super().__init__(kf_dataset)
        self.lerp_fields = lerp_fields
        self.alerp_fields = alerp_fields
        self.slerp_fields = slerp_fields
        self.kf_dataset = kf_dataset
        self.alerp_shortest_path = alerp_shortest_path
        self.slerp_spin = slerp_spin
        self.slerp_shortest_path = slerp_shortest_path
        self.reset()

    def reset(self) -> None:
        self.kf_iter = iter(self.kf_dataset)
        self.cur_kf = next(self.kf_iter)

    def eval(self, t: float) -> Optional[Tuple[MotionDataSample, KeyframeMotionDataSample]]:
        assert (
            t >= self.cur_kf.t0
        ), f"t = {t} < time of current keyframe {self.cur_kf.t0}, t must be monotonically increasing"
        while t > self.cur_kf.t1:
            try:
                self.cur_kf = next(self.kf_iter)
            except StopIteration:
                return None
        kf = self.cur_kf
        alpha = (t - kf.t0) / kf.dt
        phase = lerp(kf.phase0, kf.phase1, alpha)
        sample = self.kf_dataset.SampleType.BaseSampleType(t, kf.q0, kf.qdot, phase)
        for field in self.lerp_fields:
            sample.q_fields[field] = lerp(kf.q0_fields[field], kf.q1_fields[field], alpha)
        for field in self.alerp_fields:
            sample.q_fields[field] = alerp(
                kf.q0_fields[field],
                kf.q1_fields[field],
                alpha,
                shortest_path=self.alerp_shortest_path,
            )
        for field in self.slerp_fields:
            sample.q_fields[field] = quaternion_slerp(
                kf.q0_fields[field],
                kf.q1_fields[field],
                alpha,
                spin=self.slerp_spin,
                shortest_path=self.slerp_shortest_path,
            )
        return sample, kf
