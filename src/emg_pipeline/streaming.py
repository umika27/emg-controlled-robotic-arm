"""Causal previous-value resampling and filtering, invariant to chunk boundaries."""
from collections import deque
from time import perf_counter
import numpy as np
from .filtering import CausalFilter, design_filter
from .windows import window_sizes


class StreamingProcessor:
    def __init__(self, predictor, confidence=None, stable_windows=None):
        self.predictor = predictor
        cfg = predictor.config
        if cfg['filter']['mode'] != 'causal':
            raise ValueError('Streaming requires a causal-trained artifact; retrain with filter.mode=causal.')
        self.fs = cfg['sampling']['target_hz']; self.step = 1000/self.fs
        self.max_gap = cfg['sampling']['max_gap_ms']
        self.n, self.hop = window_sizes(self.fs, **cfg['window'])
        self.filter = CausalFilter(design_filter(self.fs, **cfg['filter']))
        self.confidence = cfg['streaming']['confidence'] if confidence is None else confidence
        self.stable = cfg['streaming']['stable_windows'] if stable_windows is None else stable_windows
        if not 0 <= self.confidence <= 1 or self.stable < 1:
            raise ValueError('Invalid confidence or stable window count.')
        self.buffer = deque(maxlen=self.n); self.votes = deque(maxlen=self.stable)
        self.pending_t = self.pending_x = self.next_t = None
        self.count = 0

    def _emit_until(self, stop):
        result = []
        while self.next_t < stop - 1e-9:
            t = self.next_t
            if t-self.pending_t > self.max_gap:
                self.buffer.clear(); self.votes.clear(); self.count = 0
                self.filter.state = None
            else:
                filtered = self.filter.process(self.pending_x[None])[0]
                self.buffer.append((t, filtered)); self.count += 1
                if len(self.buffer) == self.n and (self.count-self.n) % self.hop == 0:
                    window = np.array([entry[1] for entry in self.buffer]).T.astype(np.float32)
                    start = perf_counter(); prediction = self.predictor.predict(window)
                    elapsed = (perf_counter()-start)*1000
                    candidate = prediction['label'] if prediction['confidence'] >= self.confidence else None
                    self.votes.append(candidate)
                    stable = len(self.votes) == self.stable and candidate is not None and all(v == candidate for v in self.votes)
                    result.append(dict(label=candidate if stable else None, confidence=prediction['confidence'],
                        status=('rest' if candidate == 1 else 'gesture') if stable else 'unknown',
                        start_ms=self.buffer[0][0], end_ms=t, window_latency_ms=self.n*self.step,
                        stability_extra_ms=(self.stable-1)*self.hop*self.step, inference_ms=elapsed))
            self.next_t += self.step
        return result

    def process_chunk(self, samples, timestamps):
        x, t = np.asarray(samples, float), np.asarray(timestamps, float)
        if x.ndim != 2 or x.shape[1] != 8 or t.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(t).all():
            raise ValueError('Expected finite samples [N,8] and timestamps [N] in milliseconds.')
        if (np.diff(t) < 0).any() or (len(t) and self.pending_t is not None and t[0] < self.pending_t):
            raise ValueError('Streaming timestamps must be nondecreasing across chunks.')
        result = []
        for sample, timestamp in zip(x, t):
            if self.pending_t is None:
                self.next_t = timestamp
            elif timestamp > self.pending_t:
                result.extend(self._emit_until(timestamp))
            self.pending_t, self.pending_x = timestamp, sample.copy()
        return result

    def flush(self):
        """Emit a pending grid sample only up to the final observed timestamp."""
        return [] if self.pending_t is None else self._emit_until(self.pending_t+1e-8)
