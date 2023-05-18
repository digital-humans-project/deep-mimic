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
        self.kf_ranges = pairwise(iter(self.kf_dataset))
        self.cur_range = next(self.kf_ranges)

    def eval(self, t: float) -> Optional[MotionDataSample]:
        f0, f1 = self.cur_range
        assert (
            t >= f0.t
        ), f"t = {t} < time of current keyframe {self.cur_range[0].t}, t must be monotonically increasing"
        while t > f1.t:
            try:
                self.cur_range = next(self.kf_ranges)
            except StopIteration:
                return None
        dt = f1.t - f0.t
        alpha = (t - f0.t) / dt
        q = lerp(f0.q, f1.q, alpha)
        qdot = lerp(f0.qdot, f1.qdot, alpha)
        return self.kf_dataset.SampleType.BaseSampleType(t, q, qdot)
