# oido streaming GPU engine (Pi 400)

Copies of the live streaming dictation code. The running copies are:
- stream_ref.py, stream_gpu.py, stream_gpu2.py, stream_incr.py: ~/oido/port/
- oido-type.py: ~/voice-chat/live-mic-stream/

Sync this repo by copying the files back into place after changing them there.
Shaders, weights, caches and binaries are not included (see ~/oido/port/gpu/).

Main changes (2026-11-09):
- stream_gpu.py: chunk matmuls use gemm12 (row-major) in place of gemm_ns/gemm_nsr, which gave NaN from layer 6.
  Then k-major input (tr_km + gemm12av4) when sAm == K and K % 4 == 0: about 19% less GPU time per chunk on the live path.
- oido-type.py: PRE_ROLL = 5 s before the VAD detection, never before the end of the previous phrase (the old 1 s dropped the first words).
