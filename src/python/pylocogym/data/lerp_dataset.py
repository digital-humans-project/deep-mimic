try:
    from itertools import pairwise  # type: ignore
except ImportError:
    from itertools import tee

    def pairwise(iterable):
        # pairwise('ABCDEFG') --> AB BC CD DE EF FG
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


from typing import Optional

from pylocogym.data.dataset import IterableKeyframeMotionDataset, MotionDataSample


class ContinuousMotionDataset:
    def __init__(self, kf_dataset: IterableKeyframeMotionDataset) -> None:
        self.kf_dataset = kf_dataset

    def eval(self, t: float) -> Optional[MotionDataSample]:
        raise NotImplementedError


def lerp(a, b, alpha):
    return a * (1 - alpha) + b * alpha


class LerpMotionDataset(ContinuousMotionDataset):
    def __init__(self, kf_dataset: IterableKeyframeMotionDataset) -> None:
        super().__init__(kf_dataset)
        self.kf_dataset = kf_dataset
        self.reset()

    def reset(self) -> None:
        self.kf_iter = iter(self.kf_dataset)
        self.cur_kf = next(self.kf_iter)

    def eval(self, t: float) -> Optional[MotionDataSample]:
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
        q = lerp(kf.q0, kf.q1, alpha)
        phase = lerp(kf.phase0, kf.phase1, alpha)
        return self.kf_dataset.SampleType.BaseSampleType(t, q, kf.qdot, phase)
