"""Streaming Oido on the Pi GPU, chunk by chunk (16 frames). Front end + subsampling in numpy for now (same as
stream_ref.py). Python keeps the attention key/value cache and the conv halo between chunks; the GPU runs the 16
blocks and the head for each chunk. Verified against stream_ref.py per chunk."""
import json, os, subprocess, sys, numpy as np
sys.path.insert(0, '/home/tarel/oido/port')
import stream_ref as S
from gpu_encoder2 import roundup, fbits

HERE = '/home/tarel/oido/port'
CACHE = os.path.join(HERE, 'cache_stream')
C_CH, LEFT, HALO = 16, 128, 15
OMAX = LEFT + C_CH - 1          # 143
NP = OMAX + C_CH                # 159 positional rows

class Plan:
    def __init__(self, n_floats_limit=8 << 20):
        self.cur = 0; self.R = {}; self.gap = 131072
    def alloc(self, name, size):
        self.cur = roundup(self.cur, 64); self.R[name] = self.cur; self.cur += size + self.gap
        return self.R[name]

def make_layout(L):
    d, H, ff, k, nl, V = L['d'], L['H'], L['ff'], L['k'], L['nl'], L['V']
    P = Plan()
    lay = []
    for li in range(nl):
        lay.append(dict(P=P.alloc(f'P{li}', NP * d), KC=P.alloc(f'KC{li}', (LEFT + C_CH) * d),
                        VC=P.alloc(f'VC{li}', (LEFT + C_CH) * d), SEQ=P.alloc(f'SEQ{li}', (HALO + C_CH + HALO) * d)))
    for name, sz in [('X', C_CH * d), ('HB', C_CH * d), ('H2', C_CH * ff), ('QKV', C_CH * 3 * d), ('Q', C_CH * d),
                     ('SC', H * C_CH * (LEFT + C_CH)), ('ATT', C_CH * d), ('CV', C_CH * d), ('LOG', C_CH * (V + 1)),
                     ('TMP', C_CH * max(d, ff)),      # 2026-11-09: residual matmul output before the add (gemm12 path)
                     ('TK', C_CH * max(d, ff))]:      # 2026-11-09: k-major copy of the matmul input (tr_km + gemm12av4)
        P.alloc(name, sz)
    return P, lay

def chunk_ops(L, P, lay, n, t0, kbase, nk, kofs):
    """Plan for one chunk of n frames. Keys in KC/VC rows [0, nk) with absolute index kbase + row; the new chunk
    rows are written at kofs (= nk - n)."""
    d, H, ff, k, V = L['d'], L['H'], L['ff'], L['k'], L['V']
    dh = d // H
    R = P.R
    X, HB, H2, QKV, Q, SC, ATT, CV, LOG = (R[x] for x in ['X', 'HB', 'H2', 'QKV', 'Q', 'SC', 'ATT', 'CV', 'LOG'])
    ops = []
    def gemm(Ab, M, mm, ob, N=None, K=None, sAm=None, epi=0, resid=None):
        # 2026-11-09: gemm12 (kit shader, weights mm['wo'] = [K][NP]) instead of gemm_ns/gemm_nsr. The gemm_ns
        # path returned NaN logits from layer 6 on; gemm12 matches stream_ref to 1.2e-4 (6 chunks, argmax 100%).
        N = mm['n'] if N is None else N; K = mm['k'] if K is None else K; sAm = K if sAm is None else sAm
        NPn = roundup(mm['n'], 12)
        if resid is None:
            out = ob
        else:   # residual: v = act(gemm) into TMP, then out = X + scale * v (in place on X, row stride N = d)
            dst, scale = resid
            out = R['TMP']
        groups = max(24, ((NPn // 12) * ((M + 3) // 4) + 255) // 256)
        if sAm == K and K % 4 == 0:
            # k-major input (2026-11-09): tr_km writes A as [K][Mp] (m contiguous); gemm12av4 reads it vec4 along m.
            # 19% less GPU time per chunk on the live path, same text (scratchpad/live_k). Rows padded to a multiple of 4.
            Mp = (M + 3) // 4 * 4
            ops.append(('tr_km', 256, max(24, (Mp * K + 255) // 256), 1, [M, K, Mp, Ab, sAm, R['TK']] + [0] * 26))
            name = 'gemm12av4o4' if N % 4 == 0 else 'gemm12av4'
            ops.append((name, 256, groups, 1, [M, N, K, NPn, R['TK'], 1, Mp, mm['wo'], mm['bo'], out, N, 1, epi] + [0] * 19))
        else:
            ops.append(('gemm12', 256, groups, 1, [M, N, K, NPn, Ab, sAm, 1, mm['wo'], mm['bo'], out, N, 1, epi] + [0] * 19))
        if resid is not None:
            ops.append(('add', 128, max(24, (M * N + 127) // 128), 1, [M * N, dst, out, fbits(scale)] + [0] * 28))
    def ew(name, total, push, local=64):
        ops.append((name, local, max(24, (total + local - 1) // local), 1, push + [0] * (32 - len(push))))
    for li, (lw, lb) in enumerate(zip(L['layers'], lay)):
        ew('ln', n, [n, d, X, HB, lw['n_ff1'], lw['n_ff1'] + d])
        gemm(HB, n, lw['ff1_1'], H2, epi=2)
        gemm(H2, n, lw['ff1_2'], X, resid=(X, 0.5))
        ew('ln', n, [n, d, X, HB, lw['n_att'], lw['n_att'] + d])
        gemm(HB, n, lw['qkv'], QKV)
        ew('split_qkv', n * 3 * d, [n, d, QKV, Q, lb['KC'], lb['VC'], kofs], local=64)
        ops.append(('ch_scores', 64, max(24, (H * n * nk + 63) // 64), 1,
                    [n, nk, H, dh, d, Q, lb['KC'], lb['P'], SC, lw['pbu'], lw['pbv'], t0, kbase, OMAX, NP] + [0] * 17))
        ew('ch_softmax', H * n, [n, nk, H, dh, d, Q, lb['KC'], lb['P'], SC, lw['pbu'], lw['pbv'], 0] + [0] * 20)
        ops.append(('ch_pv', 64, max(24, (H * n * (dh // 4) + 63) // 64), 1,
                    [n, nk, H, dh, d, lb['VC'], 0, 0, SC, ATT] + [0] * 22))
        gemm(ATT, n, lw['out'], X, resid=(X, 1.0))
        ew('ln', n, [n, d, X, HB, lw['n_conv'], lw['n_conv'] + d])
        gemm(HB, n, lw['pw1'], H2)
        ew('glu', n * d, [n, d, H2, lb['SEQ'] + HALO * d], local=128)
        ops.append(('ch_dw_silu', 64, max(24, (n * d + 63) // 64), 1,
                    [n, d, k, lb['SEQ'], CV, lw['dw_w'], lw['dw_b']] + [0] * 25))
        gemm(CV, n, lw['pw2'], X, resid=(X, 1.0))
        ew('ln', n, [n, d, X, HB, lw['n_ff2'], lw['n_ff2'] + d])
        gemm(HB, n, lw['ff2_1'], H2, epi=2)
        gemm(H2, n, lw['ff2_2'], X, resid=(X, 0.5))
        ew('ln', n, [n, d, X, X, lw['n_out'], lw['n_out'] + d])
    gemm(X, n, L['head'], LOG)
    return ops

class StreamGPU:
    def __init__(self, cache=CACHE, serve=True):
        self.cache = cache
        self.L = json.load(open(os.path.join(cache, 'layout.json')))
        self.W = np.fromfile(os.path.join(cache, 'weights.bin'), dtype='<f4')
        self.P, self.lay = make_layout(self.L)
        self.size = self.P.cur
        self.env = dict(os.environ, LD_LIBRARY_PATH='/home/tarel/opt/pios-vulkan/lib',
                        VK_ICD_FILENAMES='/home/tarel/opt/pios-vulkan/icd/broadcom_icd.json')
        self.req = os.path.join(cache, 'req'); os.makedirs(self.req, exist_ok=True)
        self.proc = None
        # per-layer positional rows (fixed for all chunks): P_li = pos_linear(pe(offset))
        self.posrows = []
        d = self.L['d']
        offs = (OMAX - np.arange(NP)).astype(np.float64)
        div = np.exp((2.0 * np.arange(d // 2)) * -(np.log(10000.0) / d))
        pe = np.zeros((NP, d)); pe[:, 0::2] = np.sin(offs[:, None] * div); pe[:, 1::2] = np.cos(offs[:, None] * div)
        pe = pe.astype(np.float32)
        rnd = lambda a, b: (a + b - 1) // b * b
        for lw in self.L['layers']:
            m = lw_pos = None
            mm = self.L['layers'][len(self.posrows)]['pos']
            Wt = self.W[mm['wo']:mm['wo'] + mm['k'] * rnd(mm['n'], 12)].reshape(mm['k'], rnd(mm['n'], 12))[:, :mm['n']]
            b = self.W[mm['bo']:mm['bo'] + mm['n']]
            self.posrows.append((pe @ Wt + b).astype(np.float32))
    def start(self):
        if self.proc is None or self.proc.poll() is not None:
            self.proc = subprocess.Popen([os.path.join(HERE, 'gpu', 'exec_server'), os.path.join(self.cache, 'weights.bin')],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         text=True, env=self.env, cwd=os.path.join(HERE, 'gpu'))
        return self
    def chunk(self, Xc, state, t0, kbase):
        """Xc: (n, d) block input. state: dict with K[li], V[li] (cached rows, abs order), H[li] (halo, 15 rows).
        Returns logits (n, V+1) and updates state."""
        d, V = self.L['d'], self.L['V']
        n = Xc.shape[0]
        nkprev = state['K'][0].shape[0]
        nk = nkprev + n
        ar = np.zeros(self.size, np.float32)
        P = self.P.R
        for li, lb in enumerate(self.lay):
            ar[lb['P']:lb['P'] + NP * d] = self.posrows[li].ravel()
            ar[lb['KC']:lb['KC'] + nkprev * d] = state['K'][li].ravel()
            ar[lb['VC']:lb['VC'] + nkprev * d] = state['V'][li].ravel()
            ar[lb['SEQ']:lb['SEQ'] + HALO * d] = state['H'][li].ravel()
        ar[P['X']:P['X'] + n * d] = Xc.ravel()
        ops = []
        for li, lb in enumerate(self.lay):
            pass
        ops = chunk_ops(self.L, self.P, self.lay, n, t0, kbase, nk, nkprev)
        with open(self.req + '/c.plan', 'w') as f:
            for name, local, gx, gy, push in ops:
                f.write(f"{name} {local} {gx} {gy} " + ' '.join(str(int(x)) for x in push[:32]) + '\n')
        ar.tofile(self.req + '/c.arena')
        self.start()
        self.proc.stdin.write(f"{self.req}/c.plan {self.req}/c.arena {self.req}/c.out 0 {self.size}\n"); self.proc.stdin.flush()
        resp = self.proc.stdout.readline()
        if not resp.startswith('done'): raise RuntimeError('gpu: ' + resp.strip())
        full = np.fromfile(self.req + '/c.out', dtype='<f4')
        logits = full[P['LOG']:P['LOG'] + n * (V + 1)].reshape(n, V + 1).copy()
        # update the state from what the GPU wrote
        keep = min(nk, LEFT)
        for li, lb in enumerate(self.lay):
            state['K'][li] = full[lb['KC']:lb['KC'] + nk * d].reshape(nk, d)[nk - keep:].copy()
            state['V'][li] = full[lb['VC']:lb['VC'] + nk * d].reshape(nk, d)[nk - keep:].copy()
            state['H'][li] = full[lb['SEQ'] + n * d:lb['SEQ'] + (n + HALO) * d].reshape(HALO, d).copy()
        return logits

def run_stream(pcm, model_cache=CACHE, nchunks=None):
    """Run the streaming encoder on the GPU chunk by chunk, using numpy for the front end."""
    M = S.load('/home/tarel/oido/models/oido_stream.tnm')
    X, T = S.frontend_X(pcm, M)
    g = StreamGPU(model_cache).start()
    d = g.L['d']; nl = g.L['nl']
    state = dict(K=[np.zeros((0, d), np.float32) for _ in range(nl)], V=[np.zeros((0, d), np.float32) for _ in range(nl)],
                 H=[np.zeros((HALO, d), np.float32) for _ in range(nl)])
    out = []; kbase = 0
    for c, t0 in enumerate(range(0, T, C_CH)):
        if nchunks is not None and c >= nchunks: break
        n = min(C_CH, T - t0)
        nkprev = state['K'][0].shape[0]
        out.append(g.chunk(X[t0:t0 + n], state, t0, t0 - nkprev))
    return np.concatenate(out), T

if __name__ == '__main__':
    import time
    pcm = np.fromfile(sys.argv[1], dtype='<i2')
    t = time.time(); lg, T = run_stream(pcm, nchunks=int(sys.argv[2]) if len(sys.argv) > 2 else None)
    print('gpu stream logits', lg.shape, 'time', round(time.time() - t, 2))
