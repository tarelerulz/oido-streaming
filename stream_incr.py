"""Incremental oido GPU streaming session (2026-11-05).

Audio arrives in blocks (feed); the front end (log-mel, two stride-2 convs, subsampling) is computed only for
rows whose inputs are complete, and each 16-frame encoder chunk runs on the GPU as soon as it is ready.
finish() processes the last rows with the end-of-utterance padding, so the tail after stop is only the
remaining chunk(s). Exact: the same arithmetic as stream_ref.frontend_X / stream_gpu2.run_stream2.
"""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stream_ref as S
import stream_gpu2 as G
from stream_gpu import chunk_ops, C_CH, LEFT

def logmel_rows(x, pre, M, t0, t1):
    """Rows t0..t1-1 of stream_ref.logmel (normalized where t < nvalid of the full utterance)."""
    n = len(pre); F = np.zeros((t1 - t0, 80), np.float32)
    for j, t in enumerate(range(t0, t1)):
        idx = 160 * t - 200 + np.arange(400)
        ok = (idx >= 0) & (idx < n)
        seg = np.zeros(400, np.float32); seg[ok] = pre[idx[ok]]
        X = np.fft.rfft(seg * M['win'], 512)
        pw = (X.real ** 2 + X.imag ** 2).astype(np.float32)
        F[j] = np.log(pw @ M['fb'] + np.float32(5.9604644775390625e-08))
    return F

def conv_rows(A, i0, i1, W, b, Fin):
    """Output rows i0..i1-1 of nemo_ref.conv3x3s2 (pad 1, stride 2), reading input rows of A (real rows 0..len(A)-1)."""
    Cin = A.shape[2]; Cout = W.shape[0]
    Fout = (Fin + 2 - 3) // 2 + 1
    n = i1 - i0
    Wr = W.reshape(Cout, Cin, 3, 3)
    r_lo = 2 * i0 - 1
    blk = np.zeros((2 * n + 1, Fin + 2, Cin), np.float32)
    for k in range(2 * n + 1):
        real = r_lo + k
        if 0 <= real < len(A):
            blk[k, 1:-1] = A[real]
    out = np.zeros((n, Fout, Cout), np.float32)
    for dt in range(3):
        for df in range(3):
            sl = blk[dt:dt + 2 * n:2, df:df + 2 * Fout:2, :].reshape(n * Fout, Cin)
            out += (sl @ Wr[:, :, dt, df].T).reshape(n, Fout, -1)
    out += b
    return np.maximum(out, 0).astype(np.float32)

class IncrSession:
    def __init__(self, g, gpu_fe=None):
        self.g = g
        self.gpu_fe = gpu_fe          # gpuport.stream_fe.GpuFront: front end on the GPU (None: numpy)
        self.gpu_rows = 16            # X rows per GPU window (16 = one chunk); the tail after stop grows with it
        if getattr(g, 'M', None) is None:
            g.M = S.load('/home/tarel/oido/models/oido_stream.tnm')
        self.M = g.M
        self.g.init_state()
        self.x = np.zeros(0, np.float32)        # pcm as float (of the samples fed so far)
        self.pre = np.zeros(0, np.float32)
        self.F = np.zeros((0, 80), np.float32)
        self.C0 = np.zeros((0, self.M['sc'] * 0 + 40, self.M['c0'][0].shape[0]), np.float32)
        self.C2 = np.zeros((0, 20, self.M['c2'][0].shape[0]), np.float32)
        self.X = np.zeros((0, self.M['d']), np.float32)
        self.t_done = 0; self.nkprev = 0; self.out = []

    def _rows_ready(self):
        n = len(self.x)
        f_ready = 0 if n < 200 else (n - 200) // 160 + 1
        c0_ready = 0 if f_ready < 2 else (f_ready - 2) // 2 + 1
        c2_ready = 0 if c0_ready < 2 else (c0_ready - 2) // 2 + 1
        return f_ready, c0_ready, c2_ready

    def _extend(self, f_target=None, final=False):
        """Compute the front-end rows that are now final (or all of them when final=True)."""
        M = self.M; n = len(self.x)
        if final:
            nvalid = n // 160; T = nvalid + 1
            fr = T
        else:
            fr = self._rows_ready()[0]
        if fr > len(self.F):
            Fn = logmel_rows(self.x, self.pre, M, len(self.F), fr)
            nv = n // 160
            for j in range(Fn.shape[0]):
                t = len(self.F) + j
                if t < nv:
                    Fn[j] = (Fn[j] - M['nmean']) / M['nstd']
                else:
                    Fn[j] = 0.0
            self.F = np.concatenate([self.F, Fn])
        # c0 rows
        if final:
            c0_target = (len(self.F) + 2 - 3) // 2 + 1
        else:
            c0_target = self._rows_ready()[1]
        if c0_target > len(self.C0):
            new = conv_rows(self.F[:, :, None], len(self.C0), c0_target, M['c0'][0], M['c0'][1], 80)
            self.C0 = np.concatenate([self.C0, new])
        if final:
            c2_target = (len(self.C0) + 2 - 3) // 2 + 1
        else:
            c2_target = self._rows_ready()[2]
        if c2_target > len(self.C2):
            new = conv_rows(self.C0, len(self.C2), c2_target, M['c2'][0], M['c2'][1], 40)
            self.C2 = np.concatenate([self.C2, new])
        if len(self.C2) > len(self.X):
            flat = self.C2[len(self.X):].transpose(0, 2, 1).reshape(-1, M['sc'] * M['f2'])
            new = ((flat @ M['sub'][0].T + M['sub'][1]) * np.float32(np.sqrt(M['d']))).astype(np.float32)
            self.X = np.concatenate([self.X, new])

    def _chunk(self, n):
        g = self.g; L = g.L; V = L['V']
        t0 = self.t_done
        Xr = self.X[t0:t0 + n]
        nk = self.nkprev + n
        xfile = g.req + '/x.arena'
        Xr.astype('<f4').tofile(xfile)
        ops = g.clear_tail_ops(n)
        ops += chunk_ops(L, g.P, g.lay, n, t0, t0 - self.nkprev, nk, self.nkprev)
        ops += g.shift_ops(n, nk)
        logits = g._send(ops, xfile, g.P.R['X'], g.P.R['LOG'], n * (V + 1)).reshape(n, V + 1)
        self.out.append(logits.copy())
        self.nkprev = min(nk, LEFT)
        self.t_done += n

    def _extend_gpu(self, final=False):
        """GPU front end: X rows that are now complete (or all rows when final), appended to self.X."""
        j0 = len(self.X)
        if final:
            T_mel = len(self.x) // 160 + 1
            T2f = (T_mel + 2 - 3) // 2 + 1; T3f = (T2f + 2 - 3) // 2 + 1
            if T3f <= j0:
                return
            a = max(0, 4 * ((4 * j0 - 11) // 4))            # window start (mel row), a multiple of 4
            out, a4 = self.gpu_fe.window(self.x, a, T_mel, final=True)
            rows = out[j0 - a4:T3f - a4]
        else:
            j_ready = self._rows_ready()[2]
            if j_ready - j0 < self.gpu_rows:          # one window per gpu_rows X rows (fixed cost per window)
                return
            a = max(0, 4 * ((4 * j0 - 11) // 4))
            b = 4 * j_ready
            assert b <= self._rows_ready()[0], (b, self._rows_ready())
            out, a4 = self.gpu_fe.window(self.x, a, b, final=False)
            rows = out[j0 - a4:j_ready - a4]
        self.X = np.concatenate([self.X, rows.astype(np.float32)])

    def feed(self, pcm):
        """Add audio (int16 array). Runs every full 16-frame chunk that is ready. Returns seconds of compute."""
        t = time.perf_counter()
        f = np.asarray(pcm, np.int16).astype(np.float32) / 32768.0
        if self.gpu_fe is not None:
            self.x = np.concatenate([self.x, f]) if len(f) else self.x
            self._extend_gpu()
            while len(self.X) - self.t_done >= C_CH:
                self._chunk(C_CH)
            return time.perf_counter() - t
        if len(f):
            new = f
            prev = self.x[-1:] if len(self.x) else None
            self.x = np.concatenate([self.x, new])
            pre_new = np.empty(len(new), np.float32)
            if prev is None:
                pre_new[0] = new[0]; pre_new[1:] = new[1:] - 0.97 * new[:-1]
            else:
                pre_new[0] = new[0] - 0.97 * prev[0]; pre_new[1:] = new[1:] - 0.97 * new[:-1]
            self.pre = np.concatenate([self.pre, pre_new])
        self._extend()
        while len(self.X) - self.t_done >= C_CH:
            self._chunk(C_CH)
        return time.perf_counter() - t

    def finish(self):
        """End of utterance: process the remaining rows with the end-of-utterance padding. Returns (logits, text)."""
        t = time.perf_counter()
        if self.gpu_fe is not None:
            self._extend_gpu(final=True)
        else:
            self._extend(final=True)
        T = len(self.X)
        while self.t_done < T:
            self._chunk(min(C_CH, T - self.t_done))
        lg = np.concatenate(self.out) if self.out else np.zeros((0, self.g.L['V'] + 1), np.float32)
        self.tail_s = time.perf_counter() - t
        return lg, self.text(lg)

    def text(self, logits):
        V = self.g.L['V']; toks = self.g.L['tokens']
        out, prev = [], -1
        for a in logits.argmax(1):
            if a != prev and a != V: out.append(toks[int(a)])
            prev = a
        s_ = ''.join(out)
        return s_[1:] if s_.startswith(' ') else s_
