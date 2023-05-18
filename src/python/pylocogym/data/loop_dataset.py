from typing import List

from pylocogym.data.dataset import IterableKeyframeMotionDataset


class LoopKeyframeMotionDataset(IterableKeyframeMotionDataset):
    def __init__(
        self,
        ds: IterableKeyframeMotionDataset,
        num_loop=-1,
        track_fields: List[str] = [],
    ) -> None:
        """
        `LoopKeyframeMotionDataset` is a wrapper around `IterableKeyframeMotionDataset` that
        loops over the dataset `num_loop` times. If `num_loop` is -1, then the dataset will
        loop forever.
        """
        super().__init__()
        self.SampleType = ds.SampleType
        self.ds = ds
        self.num_loop = num_loop
        self.track_fields = track_fields

    def __iter__(self):
        i = 0
        last_state = {k: 0 for k in self.track_fields}
        t = 0
        while self.num_loop < 0 or i < self.num_loop:
            last_kf = None
            for kf in self.ds:
                kf.t0 += t
                for k, v in last_state.items():
                    kf.q0_fields[k] += v
                    kf.q1_fields[k] += v
                last_kf = kf
                yield kf
            if last_kf is not None:
                t = last_kf.t1
                for k in self.track_fields:
                    last_state[k] = last_kf.q1_fields[k]  # type: ignore
            i += 1
