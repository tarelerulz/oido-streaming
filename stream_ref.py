"""Float32 numpy reference of Oido's streaming CTC encoder (tasr_nemo.c stream path), chunk by chunk.
Chunk C=16 frames; attention keys = previous 128 frames + current chunk (no mask inside the chunk);
positional offset = query_abs - key_abs in [-(C-1), OMAX]; depthwise conv sees 15 past GLU frames, zeros after the chunk.
Front end: whole-utterance conv/sub (equal to streaming, the convs only look at already-available mel rows at the end)."""
import sys, numpy as np
sys.path.insert(0, '/home/tarel/oido/port')
from nemo_ref import Blob, conv3x3s2, layernorm, silu, dwconv

C_CH, LEFT = 16, 128
NL_CH = LEFT // C_CH           # cached chunks
OMAX = NL_CH * C_CH + C_CH - 1

def load(model):
    Bl = Blob(model); hd = Bl.hd
    d, H, ff, k, nl, sc, V, bits = hd[1], hd[2], hd[3], hd[4], hd[5], hd[6], hd[7], hd[8]
    Bl.align()
    win = Bl.f32(400); fb = Bl.f32(257 * 80).reshape(257, 80)
    c0_w = Bl.f32(sc * 9).reshape(sc, 9); c0_b = Bl.f32(sc)
    f1 = (80 + 2 - 3) // 2 + 1; f2 = (f1 + 2 - 3) // 2 + 1
    c2 = Bl.qlin(sc, sc * 9, 8); sub = Bl.qlin(d, sc * f2, 8)
    layers = []
    for _ in range(nl):
        L = {}
        L['n_ff1'] = (Bl.f32(d), Bl.f32(d)); L['ff1_1'] = Bl.qlin(ff, d, 8); L['ff1_2'] = Bl.qlin(d, ff, 8)
        L['n_att'] = (Bl.f32(d), Bl.f32(d)); L['qkv'] = Bl.qlin(3 * d, d, 8); L['out'] = Bl.qlin(d, d, 8)
        L['pos'] = Bl.qlin(d, d, 8); L['pbu'] = Bl.f32(d); L['pbv'] = Bl.f32(d)
        L['n_conv'] = (Bl.f32(d), Bl.f32(d)); L['pw1'] = Bl.qlin(2 * d, d, 8)
        L['dw_w'] = Bl.f32(d * k).reshape(d, k); L['dw_b'] = Bl.f32(d); L['pw2'] = Bl.qlin(d, d, 8)
        L['n_ff2'] = (Bl.f32(d), Bl.f32(d)); L['ff2_1'] = Bl.qlin(ff, d, 8); L['ff2_2'] = Bl.qlin(d, ff, 8)
        L['n_out'] = (Bl.f32(d), Bl.f32(d))
        layers.append(L)
    head = Bl.qlin(V + 1, d, 8)
    Bl.align(); p = Bl.p
    for _ in range(V):
        n = Bl.b[p]; p += 1 + n
    Bl.p = p
    nmean = Bl.f32(80); nstd = Bl.f32(80)
    return dict(d=d, H=H, ff=ff, k=k, nl=nl, sc=sc, V=V, f2=f2, win=win, fb=fb, c0=(c0_w, c0_b), c2=c2, sub=sub,
                layers=layers, head=head, nmean=nmean, nstd=nstd)

def logmel(pcm, M):
    n = len(pcm); x = pcm.astype(np.float32) / 32768.0
    pre = np.empty(n, np.float32); pre[0] = x[0]; pre[1:] = x[1:] - 0.97 * x[:-1]
    nvalid = n // 160; T = nvalid + 1
    F = np.zeros((T, 80), np.float32)
    for t in range(T):
        idx = 160 * t - 200 + np.arange(400)
        ok = (idx >= 0) & (idx < n)
        seg = np.zeros(400, np.float32); seg[ok] = pre[idx[ok]]
        X = np.fft.rfft(seg * M['win'], 512)
        pw = (X.real ** 2 + X.imag ** 2).astype(np.float32)
        F[t] = np.log(pw @ M['fb'] + np.float32(5.9604644775390625e-08))
    F[:nvalid] = (F[:nvalid] - M['nmean']) / M['nstd']
    F[nvalid] = 0.0
    return F, T

def frontend_X(pcm, M):
    d = M['d']
    F, Tm = logmel(pcm, M)
    c0_w, c0_b = M['c0']
    c0 = np.maximum(conv3x3s2(F.reshape(Tm, 80, 1), c0_w, c0_b), 0)
    c2 = np.maximum(conv3x3s2(c0, M['c2'][0], M['c2'][1]), 0)
    T = c2.shape[0]
    flat = c2.transpose(0, 2, 1).reshape(T, M['sc'] * M['f2'])
    X = ((flat @ M['sub'][0].T + M['sub'][1]) * np.float32(np.sqrt(d))).astype(np.float32)
    return X, T

def encode_stream(pcm, M, max_chunks=None):
    d, H, ff, k = M['d'], M['H'], M['ff'], M['k']
    dh = d // H; halo = k // 2
    F, Tm = logmel(pcm, M)
    c0_w, c0_b = M['c0']
    c0 = np.maximum(conv3x3s2(F.reshape(Tm, 80, 1), c0_w, c0_b), 0)
    c2 = np.maximum(conv3x3s2(c0, M['c2'][0], M['c2'][1]), 0)
    T = c2.shape[0]
    flat = c2.transpose(0, 2, 1).reshape(T, M['sc'] * M['f2'])
    X = ((flat @ M['sub'][0].T + M['sub'][1]) * np.float32(np.sqrt(d))).astype(np.float32)
    # positional rows: row r <-> offset OMAX - r (offset = query_abs - key_abs)
    offs = (OMAX - np.arange(OMAX + C_CH)).astype(np.float64)
    div = np.exp((2.0 * np.arange(d // 2)) * -(np.log(10000.0) / d))
    pe = np.zeros((len(offs), d)); pe[:, 0::2] = np.sin(offs[:, None] * div); pe[:, 1::2] = np.cos(offs[:, None] * div)
    pe = pe.astype(np.float32)
    P = [pe @ L['pos'][0].T + L['pos'][1] for L in M['layers']]
    Kc = [np.zeros((0, d), np.float32) for _ in M['layers']]
    Vc = [np.zeros((0, d), np.float32) for _ in M['layers']]
    Hc = [np.zeros((halo, d), np.float32) for _ in M['layers']]
    kbase = 0                                    # absolute frame index of cache row 0
    logits = []
    nchunks = 0
    for t0 in range(0, T, C_CH):
        n = min(C_CH, T - t0)
        if max_chunks is not None and nchunks >= max_chunks: break
        nchunks += 1
        x = X[t0:t0 + n].copy()
        q_abs = np.arange(t0, t0 + n)
        for li, L in enumerate(M['layers']):
            # FF1 (half step)
            h = silu(layernorm(x, *L['n_ff1']) @ L['ff1_1'][0].T + L['ff1_1'][1])
            x = x + 0.5 * (h @ L['ff1_2'][0].T + L['ff1_2'][1])
            # MHSA with chunked keys: cache (previous frames) + this chunk
            y = layernorm(x, *L['n_att'])
            qkv = y @ L['qkv'][0].T + L['qkv'][1]
            q = qkv[:, :d].reshape(n, H, dh); kk = qkv[:, d:2 * d]; vv = qkv[:, 2 * d:]
            Kc[li] = np.concatenate([Kc[li], kk]); Vc[li] = np.concatenate([Vc[li], vv])
            keys_abs = kbase + np.arange(Kc[li].shape[0])
            K = Kc[li].reshape(-1, H, dh); V = Vc[li].reshape(-1, H, dh)
            qu = q + L['pbu'].reshape(H, dh)[None]; qv = q + L['pbv'].reshape(H, dh)[None]
            Pl = P[li].reshape(-1, H, dh)
            att = np.zeros((n, H, dh), np.float32)
            scale = np.float32(1.0 / np.sqrt(dh))
            for hh in range(H):
                ac = qu[:, hh] @ K[:, hh].T                               # [n][keys]
                off = q_abs[:, None] - keys_abs[None, :]                  # query - key
                rows = OMAX - off                                          # positional row index
                bd = np.take_along_axis(qv[:, hh] @ Pl[:, hh].T, np.clip(rows, 0, Pl.shape[0] - 1), axis=1)
                s = (ac + bd) * scale
                s = s - s.max(1, keepdims=True)
                w = np.exp(s); w = w / w.sum(1, keepdims=True)
                att[:, hh] = w @ V[:, hh]
            x = x + (att.reshape(n, d) @ L['out'][0].T + L['out'][1])
            # conv module: GLU of this chunk, 15 past GLU frames as history, zeros after the chunk
            y = layernorm(x, *L['n_conv'])
            z = y @ L['pw1'][0].T + L['pw1'][1]
            g = z[:, :d] / (1.0 + np.exp(-np.clip(z[:, d:], -80, 80)))
            seq = np.concatenate([Hc[li], g, np.zeros((halo, d), np.float32)])
            cv = np.zeros((n, d), np.float32) + L['dw_b']
            for j in range(k):
                cv += seq[j:j + n] * L['dw_w'][:, j][None, :]
            cv = silu(cv)
            Hc[li] = seq[n:n + halo].copy()
            x = x + (cv @ L['pw2'][0].T + L['pw2'][1])
            # FF2 + final LN
            h = silu(layernorm(x, *L['n_ff2']) @ L['ff2_1'][0].T + L['ff2_1'][1])
            x = layernorm(x + 0.5 * (h @ L['ff2_2'][0].T + L['ff2_2'][1]), *L['n_out'])
        logits.append(x @ M['head'][0].T + M['head'][1])
        # keep only the last LEFT frames of keys/values
        keep = min(Kc[0].shape[0], LEFT)   # C code keeps the previous 128 frames before the next chunk (144 keys incl. the new chunk)
        for li in range(len(M['layers'])):
            Kc[li] = Kc[li][-keep:]; Vc[li] = Vc[li][-keep:]
        kbase = kbase + (keys_abs.shape[0] - keep)
    return np.concatenate(logits).astype(np.float32), dict(T=T, chunks=nchunks)

if __name__ == '__main__':
    import time
    M = load('/home/tarel/oido/models/oido_stream.tnm')
    pcm = np.fromfile(sys.argv[1], dtype='<i2')
    t = time.time(); lg, info = encode_stream(pcm, M); print('frames', lg.shape, info, 'time', round(time.time() - t, 2), 's')
    V = M['V']; am = lg.argmax(1); out, prev = [], -1
    for a in am:
        if a != prev and a != V: out.append(a)
        prev = a
    print('ids', out[:12])
    if len(sys.argv) > 2:
        ref = np.fromfile(sys.argv[2], dtype='<f4').reshape(-1, V + 1)
        n = min(len(ref), len(lg))
        print('vs C stream logits: argmax agree', float((ref[:n].argmax(1) == lg[:n].argmax(1)).mean()), 'max diff', float(np.abs(ref[:n] - lg[:n]).max()))
