"""Streaming Oido on the GPU with the state kept on the GPU (persistent arena in gpu/exec_server2).
Per chunk: only the 16 new input frames are sent; the logits come back. Key/value history and conv halo
are shifted on the GPU at the end of each chunk (through a temp region, so the copies never overlap).
Front end still in numpy (stream_ref.frontend_X). Verified against stream_ref.py."""
import json, os, subprocess, sys, numpy as np
sys.path.insert(0, '/home/tarel/oido/port')
import stream_ref as S
from stream_gpu import make_layout, chunk_ops, StreamGPU, C_CH, LEFT, HALO, NP, OMAX

class StreamGPU2(StreamGPU):
    def __init__(self, cache=None):
        from stream_gpu import CACHE
        super().__init__(cache or CACHE)
        # temp region for the shifts (largest copy: 128 rows of K or V)
        self.TMP = self.P.alloc('TMP', LEFT * self.L['d'])
        self.size = self.P.cur
        self.proc2 = None
        self.posdone = False
    def start2(self):
        if self.proc2 is None or self.proc2.poll() is not None:
            self.proc2 = subprocess.Popen([os.path.join(self.L_dir(), 'exec_server2'), os.path.join(self.cache, 'weights.bin')],
                                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                          text=True, env=self.env, cwd=self.L_dir())
            self.posdone = False
        return self
    def L_dir(self):
        return '/home/tarel/oido/port/gpu'
    def _send(self, plan_ops, arena_file, arena_off, out_off, out_n):
        pf = self.req + '/s.plan'
        with open(pf, 'w') as f:
            for name, local, gx, gy, push in plan_ops:
                f.write(f"{name} {local} {gx} {gy} " + ' '.join(str(int(x)) for x in push[:32]) + '\n')
        self.start2()
        self.proc2.stdin.write(f"{pf} {arena_file} {arena_off} {self.req}/s.out {out_off} {out_n}\n"); self.proc2.stdin.flush()
        resp = self.proc2.stdout.readline()
        if not resp.startswith('done'): raise RuntimeError('gpu2: ' + resp.strip())
        return np.fromfile(self.req + '/s.out', dtype='<f4') if out_n else None
    def init_state(self):
        """Write the positional rows once and zero the history regions (new utterance)."""
        d = self.L['d']
        ar = np.zeros(self.P.R['P0'] + 0 + 1, np.float32) if False else None
        ops = []
        for li, lb in enumerate(self.lay):
            ops.append(('zero_f', 64, max(24, (NP * d + 63) // 64), 1, [NP * d, lb['P']] + [0] * 30))
        # positional rows are written from numpy into the arena file (full write once)
        full = np.zeros(self.size, np.float32)
        for li, lb in enumerate(self.lay):
            full[lb['P']:lb['P'] + NP * d] = self.posrows[li].ravel()
        full.tofile(self.req + '/init.arena')
        zero_ops = []
        for li, lb in enumerate(self.lay):
            for key, cnt in [('KC', (LEFT + C_CH) * d), ('VC', (LEFT + C_CH) * d), ('SEQ', (HALO + C_CH + HALO) * d)]:
                zero_ops.append(('zero_f', 64, max(24, (cnt + 63) // 64), 1, [cnt, lb[key]] + [0] * 30))
        self._send(zero_ops, self.req + '/init.arena', 0, 0, 0)
        self.posdone = True
    def shift_ops(self, n, nk):
        """End-of-chunk: keep the last LEFT key rows at the front; keep the last HALO GLU rows; zero the unused SEQ rows."""
        d = self.L['d']; ops = []
        keep = min(nk, LEFT)
        for lb in self.lay:
            if nk > LEFT:
                src = (nk - keep) * d
                for key in ['KC', 'VC']:
                    ops.append(('copy_f', 64, max(24, (keep * d + 63) // 64), 1, [keep * d, lb[key] + src, self.TMP] + [0] * 29))
                    ops.append(('copy_f', 64, max(24, (keep * d + 63) // 64), 1, [keep * d, self.TMP, lb[key]] + [0] * 29))
            ops.append(('copy_f', 64, max(24, (HALO * d + 63) // 64), 1, [HALO * d, lb['SEQ'] + n * d, self.TMP] + [0] * 29))
            ops.append(('copy_f', 64, max(24, (HALO * d + 63) // 64), 1, [HALO * d, self.TMP, lb['SEQ']] + [0] * 29))
            rest = (HALO + C_CH + HALO - (HALO + n)) * d
            ops.append(('zero_f', 64, max(24, (rest + 63) // 64), 1, [rest, lb['SEQ'] + (HALO + n) * d] + [0] * 30))
        return ops
    def clear_tail_ops(self, n):
        d = self.L['d']; ops = []
        rest = (HALO + C_CH + HALO - (HALO + n)) * d
        for lb in self.lay:
            ops.append(('zero_f', 64, max(24, (rest + 63) // 64), 1, [rest, lb['SEQ'] + (HALO + n) * d] + [0] * 30))
        return ops
    def run_stream2(self, pcm):
        X, T = S.frontend_X(pcm, self.M)
        d, V = self.L['d'], self.L['V']
        self.init_state()
        out = []; nkprev = 0
        for t0 in range(0, T, C_CH):
            n = min(C_CH, T - t0)
            nk = nkprev + n
            # input: only this chunk's frames go into the arena X region
            xr = self.P.R['X']
            xfile = self.req + '/x.arena'
            X[t0:t0 + n].astype('<f4').tofile(xfile)
            ops = self.clear_tail_ops(n)
            ops += chunk_ops(self.L, self.P, self.lay, n, t0, t0 - nkprev, nk, nkprev)
            ops += self.shift_ops(n, nk)
            logits = self._send(ops, xfile, xr, self.P.R['LOG'], n * (V + 1)).reshape(n, V + 1)
            out.append(logits.copy())
            nkprev = min(nk, LEFT)
        return np.concatenate(out), T
    def run(self, pcm):
        if getattr(self, 'M', None) is None:
            self.M = S.load('/home/tarel/oido/models/oido_stream.tnm')
        return self.run_stream2(pcm)
    def text(self, logits):
        V = self.L['V']; toks = self.L['tokens']
        out, prev = [], -1
        for a in logits.argmax(1):
            if a != prev and a != V: out.append(toks[int(a)])
            prev = a
        s_ = ''.join(out)
        return s_[1:] if s_.startswith(' ') else s_
    def transcribe(self, pcm):
        lg, _ = self.run(np.asarray(pcm, np.int16))
        return self.text(lg)

if __name__ == '__main__':
    import time
    pcm = np.fromfile(sys.argv[1], dtype='<i2')
    g = StreamGPU2()
    t = time.time(); lg, T = g.run(pcm); print('gpu2 stream', lg.shape, 'time', round(time.time() - t, 2), 's')
