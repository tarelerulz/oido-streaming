# oido streaming GPU engine (Pi 400)

**Reference code, not a standalone install.** These are copies of the scripts the author runs for live dictation. They expect the folder layout of the author's Pi (`~/oido/port/`, `~/voice-chat/live-mic-stream/`) and need the GPU runner, shaders and model weights from [gpu-port-kit](https://github.com/tarelerulz/gpu-port-kit) and [oido](https://github.com/tarelerulz/oido), which are not included here. Read them to see how the streaming path is built; the story of the GPU work is in [OIDO_GPU_STORY.md](https://github.com/tarelerulz/gpu-port-kit/blob/main/OIDO_GPU_STORY.md).

Copies of the live streaming dictation code. The running copies are:
- stream_ref.py, stream_gpu.py, stream_gpu2.py, stream_incr.py: ~/oido/port/
- oido-type.py: ~/voice-chat/live-mic-stream/

Sync this repo by copying the files back into place after changing them there.
Shaders, weights, caches and binaries are not included (see ~/oido/port/gpu/).

Main changes (2026-11-09):
- stream_gpu.py: chunk matmuls use gemm12 (row-major) in place of gemm_ns/gemm_nsr, which gave NaN from layer 6.
  Then k-major input (tr_km + gemm12av4) when sAm == K and K % 4 == 0: about 19% less GPU time per chunk on the live path.
- oido-type.py: PRE_ROLL = 5 s before the VAD detection, never before the end of the previous phrase (the old 1 s dropped the first words).

Sync 2026-11-19 (oido-type.py): optional hybrid CPU+GPU split of long phrases, off unless `OIDO_HYBRID=1` (used with the full-GPU
engine). A phrase of 4.5 s or more is cut at a real pause; the GPU transcribes the head while a resident `tasr_cli --serve` on
cores 1-3 transcribes the tail; any CPU-side problem falls back to the GPU for that phrase. Measured on a Pi 400: 10 s clip
488 -> 390 ms, 15 s 817 -> 501 ms, same text. The real gain in live dictation is small (early transcription already hides most of
the wait) and word quality in daily use is not judged yet. A standalone, tested version of the class and the thread-pool fix it
needs are in a draft pull request on the oido repo (tarelerulz/oido #16).
