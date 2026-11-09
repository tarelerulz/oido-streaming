#!/home/tarel/voice-chat/venv-asr/bin/python
"""oido-type.py — TRIAL (2026-11-02): dictation into a tmux pane with Lokutor's Oido (nemo8 + LM + punctuation).

A copy of whistle-type.py: the same Silero VAD phrase cutting, PREROLL (audio kept from before the speech is
noticed, so soft first words survive), voice commands, "stop dictation", nano checks and clip saving. Only the
recogniser changed: each phrase goes to the tasr_cli_a72 binary (nemo8.tnm + nemo_lm.tlm, ~0.5 s per 10 s).
Punctuation comes from the keep-loaded oido-punct server (~/bin/oido-punct.py; ~0.17 s per phrase).

Usage:
  oido-type.py --target <pane>
  oido-type.py --target <pane> --wav test.wav   (feeds a file in real time)
"""
import argparse
import importlib.util
import json
import os
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fcm_type", str(HERE / "fcm-stream-type.py"))
et = importlib.util.module_from_spec(spec)
spec.loader.exec_module(et)
vt = et.vt
sys.path.insert(0, str(HERE))
import dictation_commands  # noqa: E402  spoken punctuation + editing, shared with the other dictation keys
NEEDLE = Path.home() / "src/needle-linux-arm64/needle"
WHISTLE = Path.home() / ".cache/cactus-needle/whistle/2.0.0/whistle.cact"
SILERO_VAD = Path("/mnt/pios-root/home/tarel/models/silero_vad.onnx")
STOP_HOLD_SECONDS = 4.0
FILLER = re.compile(r"\b(?:u+m+|u+h+|h+m+|m+h*m+|m+-?hm+|mm+-?hmm+|a+h+|e+r+m*)\b[,.]?\s*", re.I)
# Live phone test 10-05: Whistle's spellings of Tarel. Real "Terrell" is left alone (Tarel, 2026-09-06).
et.NAME_FIXES += [(re.compile(r"\b(?:Terral|Terrile)\b", re.I), "Tarel"),
                  (re.compile(r"\bGentude\b", re.I), "Gentoo")]


# Whistle's spellings of "stop dictation" (2026-10-08: "STOT DICTATION." twice got typed into an email):
# STOT TICTATION, STOP DICATION, Stopdication... Same pattern as ~/src/wake-word/train.py REAL_STOP;
# "stop the station", "stop and listen", "sophistication" don't match.
MISHEARD_STOP = re.compile(r"^s?t[oa][pt]\s*[dt]i[ck]{0,2}t?ation$")


def is_stop_any(spoken):
    return et.is_stop(spoken) or bool(MISHEARD_STOP.match(spoken)) or bool(MISHEARD_STOP.match(spoken.replace(" ", "")))


def split_stop(spoken):
    """'... some words stop dictation' -> ('... some words', True)."""
    if is_stop_any(spoken):
        return "", True
    words = spoken.split()
    for n in (3, 2):
        if len(words) > n and is_stop_any(" ".join(words[-n:])):
            return " ".join(words[:-n]), True
    return spoken, False


# The tiny wake-word model (~/bin/kws_np.py) also listens for "stop dictation" in SHORT phrases (<= 3 s):
# if it is >= KWS_STOP sure, dictation stops even when Whistle misheard the words. Off: WHISTLE_KWS_STOP=off.
KWS_STOP = float(os.environ.get("WHISTLE_KWS_STOP_SURE", "0.7"))
_kws = None


def kws_says_stop(samples):
    global _kws
    if os.environ.get("WHISTLE_KWS_STOP", "on").lower() in ("off", "0", "no") or len(samples) > 3 * RATE:
        return None
    try:
        if _kws is None:
            sys.path.insert(0, str(Path.home() / "bin"))
            import kws_np
            _kws = kws_np.KWS()
            if os.environ.get("WHISTLE_KWS_GPU", "on").lower() not in ("off", "0", "no"):
                try:                                   # GPU version (~/src/kws-gpu): same answers, ~19x less CPU
                    sys.path.insert(0, str(Path.home() / "src/kws-gpu"))
                    import kws_gpu
                    _kws = kws_gpu.KWSGPU()
                    log("wake-word model for stop: GPU")
                except Exception as e:
                    log(f"wake-word model for stop: CPU (GPU not available: {e})")
        try:
            return _kws.utterance(samples)[1]
        except Exception as e:                         # a GPU failure mid-session: fall back to the CPU
            import kws_np
            log(f"wake-word model: GPU failed, CPU from now on: {e}")
            _kws = kws_np.KWS()
            return _kws.utterance(samples)[1]
    except Exception as e:
        log(f"wake-word model failed: {e}")
        return None


OIDO_CLI = Path.home() / "oido/host/tasr_cli_a72"
OIDO_MODEL = Path.home() / "oido/models/nemo8.tnm"
OIDO_LM = Path.home() / "oido/models/nemo_lm.tlm"
OIDO_PUNCT = Path.home() / "bin/oido-punct.py"
OIDO_SPEECH_KEEP = Path.home() / "bin/oido-speech-keep.py"


OIDO_GPU_DIR = Path.home() / "oido/port"   # gpu_encoder.py: nemo8 encoder on the Pi GPU (2026-11-03)

# 2026-11-08: while heart reads aloud (email, replay), the mic would hear the speaker and type it.
# read-buffer-aloud.py writes the mpv pid to these files; dictation ignores the mic while one is alive,
# plus a short tail after it stops (room echo).
READ_ALOUD_PIDS = [Path.home() / ".local/share/read-aloud/reading.pid",
                   Path.home() / ".local/share/read-aloud/replay.pid"]
TTS_TAIL = 0.6          # seconds after the reader stops before the mic counts again
_tts_last_busy = 0.0


def tts_playing():
    """True while a read-aloud mpv is playing (or just stopped). Stale pid files are ignored."""
    global _tts_last_busy
    busy = False
    for pid_file in READ_ALOUD_PIDS:
        try:
            pid = int(pid_file.read_text().strip())
            if b"mpv" in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[0]:
                busy = True
        except (OSError, ValueError):
            pass
    now = time.time()
    if busy:
        _tts_last_busy = now
        return True
    return now - _tts_last_busy < TTS_TAIL


class OidoCLI:
    """One phrase -> text with the Oido CLI (nemo8 + LM), then punctuation from the keep-loaded punct server.
    If the server is not running, the punctuation model is loaded here once (as in voice-type.py).
    gpu=True: the encoder runs on the GPU (gpu_encoder.py, resident server); the CPU does features and decoding."""
    def __init__(self, gpu=False, gpu_full=False, gpu_stream=False):
        self.gpu = None
        if gpu_stream:   # streaming model, whole front end + 16 blocks on the GPU in 16-frame chunks (stream_gpu2.py)
            sys.path.insert(0, str(OIDO_GPU_DIR))
            import stream_gpu2
            self.gpu = stream_gpu2.StreamGPU2().start2()
            self.gpu.M = None
            log("oido streaming GPU: server started")
        elif gpu or gpu_full:
            sys.path.insert(0, str(OIDO_GPU_DIR))
            if gpu_full:   # whole front end + encoder on the GPU: kit encoder (~/gpu-port-kit, 2026-11-05; replaces gpu_encoder2.py)
                sys.path.insert(0, str(Path.home() / "gpu-port-kit"))
                from gpuport.nemo8 import Nemo8Encoder
                self.gpu = Nemo8Encoder().start()
                log("oido full GPU (kit encoder, front end + encoder): server started")
            else:
                import gpu_encoder
                self.gpu = gpu_encoder.GpuEncoder().start()
                log("oido GPU encoder: server started")
        spec = importlib.util.spec_from_file_location("oido_punct", str(OIDO_PUNCT))
        self.punct_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.punct_mod)
        self.punctuator = None
        # keep-loaded speech server (~/bin/oido-speech-keep.py): model stays in memory between phrases
        spec2 = importlib.util.spec_from_file_location("oido_speech_keep", str(OIDO_SPEECH_KEEP))
        self.speech_keep = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(self.speech_keep)

    def transcribe(self, samples):
        if self.gpu is not None:
            pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
            try:
                text = self.gpu.transcribe(pcm).strip()
            except Exception as e:  # GPU server trouble: fall back to the CPU path for this phrase
                log(f"oido GPU failed, CPU for this phrase: {e}")
                text = None
            if text is not None:
                if not text:
                    return ""
                try:
                    return self.punct_mod.via_server(text)
                except OSError:
                    if self.punctuator is None:
                        self.punctuator = self.punct_mod.make_punctuator()
                    return self.punct_mod.punctuate(self.punctuator, text)
        with tempfile.NamedTemporaryFile(suffix=".wav", dir="/dev/shm") as f:
            with wave.open(f.name, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
                w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
            text = None
            try:
                text = self.speech_keep.transcribe(f.name)
            except OSError:
                text = None
            if text is None:  # no server yet: start it for later phrases, do this one the old way
                try:
                    self.speech_keep.start_server()
                except OSError:
                    pass
                out = subprocess.run([str(OIDO_CLI), str(OIDO_MODEL), "--lm", str(OIDO_LM), f.name],
                                     capture_output=True, text=True)
                line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
                text = line.split("\t", 1)[1].strip() if "\t" in line else ""
                if not text:
                    log(f"oido failed: {out.stderr.strip()[-200:]}")
                    return ""
            text = text.strip()
            if not text:  # silence or nothing recognised
                return ""
        try:
            return self.punct_mod.via_server(text)
        except OSError:
            if self.punctuator is None:
                self.punctuator = self.punct_mod.make_punctuator()
            return self.punct_mod.punctuate(self.punctuator, text)

    def punctuate(self, text):
        """Punctuation for text that came from elsewhere (the incremental GPU stream), 2026-11-08."""
        text = (text or "").strip()
        if not text:
            return ""
        try:
            return self.punct_mod.via_server(text)
        except OSError:
            if self.punctuator is None:
                self.punctuator = self.punct_mod.make_punctuator()
            return self.punct_mod.punctuate(self.punctuator, text)


class Whistle:
    """One phrase -> text with the standalone needle program (a fresh process per phrase, ~0.1 s load)."""
    def __init__(self, depth, threads):
        self.cmd = [str(NEEDLE), "--model", str(WHISTLE), "--audio-language", "en",
                    "--audio-depth", str(depth), "--threads", str(threads)]

    def transcribe(self, samples):
        with tempfile.NamedTemporaryFile(suffix=".wav", dir="/dev/shm") as f:
            with wave.open(f.name, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
                w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
            out = subprocess.run(self.cmd + ["--audio", f.name], capture_output=True, text=True)
        try:
            return json.loads(out.stdout)["text"]
        except (ValueError, KeyError):
            log(f"needle failed: {out.stderr.strip()[-200:]}")
            return ""

class KeptWhistle:
    """--kept (2026-10-10, default): the same Whistle model, loaded once in this process (the Python engine) instead of
    a fresh needle program per phrase. 32/32 real dictation clips gave the same text; 130-170 ms faster per phrase
    (short phrases 333 -> 202 ms). Shares the model with the live preview; `lock` makes them take turns (the engine
    is one process-wide model; tested: transcribing between stream passes leaves both results unchanged)."""
    def __init__(self):
        import glob
        sys.path += glob.glob(str(Path.home() / "venvs/whistle-stream/lib/python3*/site-packages"))
        os.environ.setdefault("NEEDLE_TELEMETRY", "0")
        import needle
        self.model = needle.Whistle()
        self.lock = threading.Lock()

    def transcribe(self, samples):
        # The same 16-bit rounding the WAV file gave the needle program: on noisy phrases a difference that small
        # changed Whistle's words (10-10 test: one 12 s clip looped "But he didn't have things at once." x4).
        pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16).astype(np.float32) / 32768
        with self.lock:
            return self.model.transcribe(pcm, language="en")["text"]

RATE = 16000
CHUNK = 1280
MIN_SILENCE = 0.6                # seconds of quiet that end a phrase
PRE_ROLL = 5.0                   # 2026-11-09: seconds of audio fed to the GPU stream before the VAD detection
EARLY = 0.25                     # start transcribing after this much quiet; type only after MIN_SILENCE
# Phrase clips (2026-10-07, Tarel: keep them so fixes can be tested on real mistakes). Each typed phrase's audio
# + Whistle's text go to CLIPS_DIR/<date>/ (wav + clips.jsonl); days older than DICTATION_KEEP_DAYS are deleted
# at start. Local only. --no-save turns it off; test runs (--wav) never save.
CLIPS_DIR = Path(os.environ.get("DICTATION_CLIPS_DIR", Path.home() / ".local/share/dictation-clips"))
KEEP_DAYS = int(os.environ.get("DICTATION_KEEP_DAYS", "7"))
PAD_BEFORE, PAD_AFTER = 0.4, 0.2  # VAD clips word starts: "stop dictation" came out "== Tation ==" (dry run 10-05)
LOG = Path.home() / ".cache/whistle-type.log"
RATE = 16000
DIM, RESET, CLEAR = et.DIM, et.RESET, et.CLEAR


def log(msg):
    try:
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a") as f:
            f.write(time.strftime("%H:%M:%S ") + msg + "\n")
    except OSError:
        pass


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="tmux pane to type into")
    ap.add_argument("--mic", default=vt.MIC_DEVICE, help="ALSA capture device")
    ap.add_argument("--wav", help="test: feed this 16 kHz WAV in real time instead of the mic")
    ap.add_argument("--gpu-stream", action="store_true", help="oido: streaming model on the Pi GPU (stream_gpu2.py)")
    ap.add_argument("--incr", action="store_true", help="with --gpu-stream: feed the GPU while you talk; the text after stop is ~0.3 s (stream_incr.py)")
    ap.add_argument("--cpu-fe", action="store_true", help="with --incr: keep the front end on the CPU (default: GPU)")
    ap.add_argument("--gpu-full", action="store_true", help="oido: whole front end and encoder on the Pi GPU (gpu_encoder2.py)")
    ap.add_argument("--gpu-encoder", action="store_true", help="oido: run the speech encoder on the Pi GPU (CPU does features/decoding)")
    ap.add_argument("--depth", type=int, default=8, help="decoder layers 2..8 (4 = same words in tests, faster; 3 breaks)")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--silence", type=float, default=MIN_SILENCE, help="seconds of quiet that end a phrase")
    ap.add_argument("--live", action="store_true", help="show words while you talk (Whistle streaming preview)")
    ap.add_argument("--live-block", type=float, default=1.0, help="seconds of audio per streaming pass (Pi: 1.0)")
    ap.add_argument("--live-window", type=float, default=5.0,
                    help="restart the live stream past this many seconds (0 = re-read the whole phrase, as before)")
    ap.add_argument("--fresh-process", action="store_true",
                    help="typed text from a fresh needle program per phrase (before 2026-10-10) instead of the loaded model")
    ap.add_argument("--no-save", action="store_true", help="don't keep phrase clips (see CLIPS_DIR)")
    ap.add_argument("--save-clips", action="store_true", help="test: save clips even with --wav")
    ap.add_argument("--early", type=float, default=EARLY,
                    help="start transcribing after this much quiet (0 = off: transcribe only after --silence)")
    return ap.parse_args()


def preview_width():
    """The live line must fit one row of the small pane, or it wraps and redraws badly."""
    import shutil
    return max(10, shutil.get_terminal_size((80, 8)).columns - 1)


TAIL_QUIET = 0.25                # seconds at the end of a live block that must hold speech to start a pass
QUIET_SHARE = 0.35              # ...and be at least this share of the phrase's loudest block
RESTART_SHARE = 0.6             # --live-window: restart the stream after a block whose tail is this quiet
RESTART = object()              # queue marker: end this stream, start a new one


class Live:
    """--live (2026-10-06): Whistle STREAMING (needle PR #168, engine 3.2.0, in-process) shows the words in this pane
    while you talk; the typed text still comes from the batch path at the pause (faster end, same words).
    Streaming re-reads the phrase so far once per block; on the Pi a 1 s block keeps up (~0.55 s per pass)."""
    def __init__(self, block, kept, window):
        self.kept = kept
        self.model = kept.model
        self.block = int(block * RATE)
        # --live-window (2026-10-10): each streaming pass re-reads everything since the stream started (0.17 s at
        # 1-3 s of speech, 1.5 s past 10 s in the 10-07..10-10 log), so long phrases kept the CPU busy and the typed
        # text waited. Past `window` seconds the stream restarts at the next quieter block (at 2x window regardless);
        # the words already shown stay. Preview only: the typed text still comes from the whole phrase.
        self.window = int(window * RATE)
        self.fed = 0                                  # samples sent to the current stream
        self.q, self.thread, self.buf, self.paused, self.quiet = None, None, [], False, True
        self.busy = False                             # a streaming pass is running
        self.ref = 0.0                                # loudest block of this phrase (speech level)

    def start(self, preroll):
        if self.thread is not None:
            self.thread.join(3)                       # the last phrase's final pass
        self.q, self.buf, self.paused, self.quiet = queue.Queue(), [preroll], False, False
        self.ref, self.fed = 0.0, 0
        self.thread = threading.Thread(target=self._run, args=(self.q,), daemon=True)
        self.thread.start()

    def feed(self, chunk):
        if self.q is None or self.paused:
            return
        self.buf.append(chunk)
        if sum(len(b) for b in self.buf) >= self.block:
            audio = np.concatenate(self.buf)
            self.ref = max(self.ref, float(np.sqrt(np.mean(audio ** 2))))
            tail = audio[-int(TAIL_QUIET * RATE):]
            if float(np.sqrt(np.mean(tail ** 2))) < QUIET_SHARE * self.ref:
                # The block ends quiet: probably the pause that ends the phrase. A pass started now would still be
                # running when the typed text needs the CPU (10-07 log: 25 of 83 phrases waited, up to 0.8 s).
                # Hold it; it goes with the next block if you keep talking.
                self.buf = [audio]
                return
            self.q.put(audio)
            self.buf = []
            self.fed += len(audio)
            if self.window and self.fed >= self.window and (
                    float(np.sqrt(np.mean(tail ** 2))) < RESTART_SHARE * self.ref or self.fed >= 2 * self.window):
                self.q.put(RESTART)
                self.fed = 0

    def end(self):
        if self.q is not None:
            self.quiet = True
            self.q.put(None)
            self.q = None

    def _run(self, q):
        lock = self.kept.lock
        held = [False]

        def take():                                   # the model is shared with the typed-text path
            lock.acquire()
            held[0], self.busy = True, True

        def give():
            if held[0]:
                held[0], self.busy = False, False
                lock.release()

        committed, ended = "", False
        try:
            while not ended:
                restart = False

                def chunks():                        # iter(q.get, None) compares numpy arrays to None
                    nonlocal ended, restart
                    while True:
                        block = q.get()
                        if block is None or block is RESTART:
                            ended, restart = block is None, block is RESTART
                            take()                    # the stream's closing pass
                            return
                        parts = [block]
                        while not q.empty():          # passes slower than a block: merge, never fall behind
                            nxt = q.get_nowait()
                            if nxt is None or nxt is RESTART:
                                ended, restart = nxt is None, nxt is RESTART
                                break
                            parts.append(nxt)
                        take()
                        yield np.concatenate(parts)
                        if ended or restart:
                            take()
                            return

                for step in self.model.stream(chunks(), language="en"):
                    give()
                    log(f"live pass {step.get('pass_ms', 0):.0f} ms at {step.get('received')} s")
                    if step.get("text"):
                        committed = (committed + " " + step["text"]).strip()
                    if not self.quiet:
                        shown = (committed + " " + DIM + (step.get("pending") or "") + RESET).strip()
                        print(f"{CLEAR}{shown[-(preview_width() + 8):]}", end="", flush=True)
                if restart:
                    log("live stream restarted")
        except Exception as e:                       # the preview must never stop dictation
            log(f"live preview failed: {e}")
        finally:
            give()

    def wait_idle(self, limit=0.8):
        """Let a running pass finish first: both engines at once (4 + 3 threads on 4 cores) slowed the typed text
        by 0.3-1.5 s in the 10-06 test; one after the other is faster."""
        t0 = time.time()
        while self.busy and time.time() < t0 + limit:
            time.sleep(0.01)
        if time.time() - t0 > 0.02:
            log(f"waited {time.time() - t0:.2f} s for the live pass")


def to_pcm16(x):
    """The same 16-bit rounding as OidoCLI.transcribe."""
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def prune_clips():
    """Delete clip days older than KEEP_DAYS."""
    import datetime
    import shutil
    cutoff = datetime.date.today() - datetime.timedelta(days=KEEP_DAYS)
    for day in CLIPS_DIR.glob("????-??-??"):
        try:
            if datetime.date.fromisoformat(day.name) < cutoff:
                shutil.rmtree(day)
        except ValueError:
            pass


def save_clip(samples, heard, why):
    """One phrase's audio + what Whistle heard, for later testing."""
    stamp = time.time()
    day = CLIPS_DIR / time.strftime("%Y-%m-%d", time.localtime(stamp))
    day.mkdir(parents=True, exist_ok=True)
    name = time.strftime("%H%M%S", time.localtime(stamp)) + f"-{int(stamp * 1000) % 1000:03d}.wav"
    with wave.open(str(day / name), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    with open(day / "clips.jsonl", "a") as f:
        f.write(json.dumps({"file": name, "seconds": round(len(samples) / RATE, 2), "heard": heard,
                            "pass": why, "engine": "whistle needle depth 8"}) + "\n")


def mic_thread(args, q):
    """Read audio in its own thread so nothing is lost while Whistle is busy."""
    for chunk in et.audio_chunks(args):
        q.put(chunk)
    q.put(None)


def main():
    args = parse_args()
    pane = args.target
    if not vt.pane_exists(pane):
        sys.exit(f"pane {pane} not found")
    expected_cmd = vt.pane_active_command(pane)
    if expected_cmd in vt.NO_DICTATION_PROGRAMS:
        vt.tmux("display-message", "-d", "6000", "-t", pane,
                f"voice-type: won't type into {expected_cmd} (keys are commands there)",
                check=False)
        sys.exit(f"pane {pane} is running '{expected_cmd}'. Open the editor (nano) first.")
    print("Loading Oido (TRIAL)...", flush=True)
    t = time.time()
    whistle = OidoCLI(gpu=args.gpu_encoder, gpu_full=args.gpu_full, gpu_stream=args.gpu_stream)
    gpu_front = None
    incr = bool(args.incr and args.gpu_stream and whistle.gpu is not None)
    if incr:
        # the early pass and batch runs would reset the GPU state of the open session (init_state): off in this mode
        args.early = 0
        sys.path.insert(0, str(OIDO_GPU_DIR))
        import stream_incr
        if not args.cpu_fe:          # front end (log-mel, convs, subsampling) on the GPU too (~/gpu-port-kit/gpuport/stream_fe.py)
            try:
                sys.path.insert(0, str(Path.home() / "gpu-port-kit"))
                from gpuport.stream_fe import GpuFront
                gpu_front = GpuFront(str(Path.home() / "oido/models/oido_stream.tnm"))
            except Exception as e:
                log(f"GPU front end failed, CPU front end: {e}")
        log("oido incremental GPU stream: on (front end " + ("GPU" if gpu_front else "CPU") + ")")
    import sherpa_onnx
    vcfg = sherpa_onnx.VadModelConfig()
    vcfg.silero_vad.model = str(SILERO_VAD)
    vcfg.silero_vad.threshold = 0.5
    vcfg.silero_vad.min_speech_duration = 0.2
    vcfg.silero_vad.min_silence_duration = args.silence
    vcfg.silero_vad.max_speech_duration = 25        # Whistle takes at most 30 s
    vcfg.sample_rate = RATE
    vad = sherpa_onnx.VoiceActivityDetector(vcfg, buffer_size_in_seconds=60)
    quick = None
    if args.early:
        qcfg = sherpa_onnx.VadModelConfig()
        qcfg.silero_vad.model = str(SILERO_VAD)
        qcfg.silero_vad.threshold = 0.5
        qcfg.silero_vad.min_speech_duration = 0.2
        qcfg.silero_vad.min_silence_duration = args.early
        qcfg.silero_vad.max_speech_duration = 25
        qcfg.sample_rate = RATE
        quick = sherpa_onnx.VoiceActivityDetector(qcfg, buffer_size_in_seconds=60)
    live = Live(args.live_block, kept, args.live_window) if args.live else None
    saving = (not args.no_save and not args.wav) or args.save_clips
    if saving:
        prune_clips()
    dictation_commands.show_card()
    # 2026-11-06: say which model is in use (on screen and as the pane title)
    if args.gpu_stream:
        model_label = "oido streaming GPU" + (" (incremental, GPU front end)" if incr and gpu_front else
                                              " (incremental, CPU front end)" if incr else "")
    elif args.gpu_full:
        model_label = "oido full GPU (nemo8)"
    elif args.gpu_encoder:
        model_label = "oido GPU encoder (CPU front end)"
    else:
        model_label = "oido CPU (nemo8 + LM)"
    print(f"Model: {model_label}", flush=True)
    try:
        import subprocess as _sp
        _sp.run(["tmux", "select-pane", "-t", pane, "-T", model_label], check=False, capture_output=True)
    except Exception:
        pass
    print(f"Ready in {time.time() - t:.1f} s. Typing into {expected_cmd} (pane {pane}).")
    vt.tmux("display-message", "-d", "3000", "-t", pane,
            f"voice-type: WHISTLE trial, typing into {expected_cmd}", check=False)
    signal.signal(signal.SIGINT, lambda *_: sys.exit("\nBye!"))

    sentence_ended = True
    typed_space = False
    end_mark = None
    typer = dictation_commands.Typer(vt, et.tidy, pane)
    held = None                       # time of a lone "stop" waiting for "dictation"

    def handle(raw, literal=False):
        """One finished phrase; returns False to stop dictation."""
        nonlocal sentence_ended, typed_space, end_mark, held
        raw = FILLER.sub("", raw).strip()   # the model's own " marks are quote commands now (dictation_commands)
        raw = re.sub(r"[_*#~^|<>\[\]{}]+", "", raw).strip()   # stray symbols Whistle sometimes adds ("_Envious")
        spoken = re.sub(r"[^\w\s]", "", raw.lower()).strip()
        if held is not None:
            t0, held = held, None
            if raw and time.time() - t0 < STOP_HOLD_SECONDS and et.is_stop("stop " + spoken):
                log("stop")
                print(f"{CLEAR}Bye!")
                return False
            if not handle("Stop", literal=True):
                return False
        if not literal and re.fullmatch(r"s?top\w*", spoken):
            held = time.time()
            log(f"held: {raw}")
            print(f"{CLEAR}{DIM}stop... (say 'dictation')", flush=True)
            return True
        if not raw:
            return True
        rest, stop = split_stop(spoken)
        if not stop and spoken.endswith("top dictation"):     # Whistle drops the s (file test 10-05)
            rest, stop = spoken[:-len("top dictation")].strip(), True
        if stop:
            if rest:
                rest_raw = " ".join(raw.split()[:len(rest.split())])
                vt.stop_if_program_changed(pane, expected_cmd)
                done = typer.phrase(rest_raw)
                if done:
                    print(f"Typed: {done}")
                    log(f"typed: {done}")
            log("stop")
            print("Bye!")
            return False
        if not vt.pane_exists(pane):
            sys.exit(f"pane {pane} is gone")
        vt.stop_if_program_changed(pane, expected_cmd)
        done = typer.phrase(raw)       # text + spoken punctuation/editing (dictation_commands.py, 2026-10-11)
        if done:
            print(f"Typed: {done}", flush=True)
            log(f"typed: {done}")
        return True

    q = queue.Queue()
    t_feed = time.time()                # --wav: the file plays in real time from here
    threading.Thread(target=mic_thread, args=(args, q), daemon=True).start()
    listening = False
    history = np.zeros(0, np.float32)   # last 40 s of audio, for padding phrases
    last_end = 0                        # 2026-11-09: sample index where the last phrase was closed (preroll stops here)
    seen = 0                            # samples given to the VAD so far
    # Early transcription (2026-10-05): a second, quick VAD (--early quiet) starts transcribing the phrase so far
    # while the normal VAD (--silence) still decides where a phrase ends. When it does, the text is usually ready;
    # if the phrase turned out longer (you kept talking), it is transcribed again. Same phrase cuts as without it.
    # (A single VAD with a typing timer typed too soon: speech restart is noticed ~0.2 s late -> "to... Create".)
    ready = None                        # (start, end, text) from the quick VAD
    sess = None                         # --incr: the open GPU stream session for this phrase
    pad_a, pad_b = int(PAD_BEFORE * RATE), int(PAD_AFTER * RATE)
    p_start = None                      # start of the phrase the normal VAD is still in

    def cut(a0, b0):
        first = seen - len(history)
        return history[max(first, a0 - pad_a) - first:min(seen, b0 + pad_b) - first]

    def run(a0, b0, why):
        seg = cut(a0, b0)
        t0 = time.time()
        raw = whistle.transcribe(seg).strip()
        took = time.time() - t0
        if live is None:
            print(f"{CLEAR}{DIM}({len(seg) / RATE:.1f} s speech -> {took:.2f} s){RESET}", end="", flush=True)
        log(f"heard ({len(seg) / RATE:.1f} s, {took:.2f} s, {why}): {raw}")
        return raw

    tts_muted = False
    while (chunk := q.get()) is not None:
        if tts_playing():               # heart is reading aloud: drop the mic audio (2026-11-08)
            if not tts_muted:
                log("mic muted while read-aloud plays")
                tts_muted = True
            continue
        if tts_muted:
            log("mic back on")
            tts_muted = False
        vad.accept_waveform(chunk)
        seen += len(chunk)
        history = np.concatenate([history, chunk])[-40 * RATE:]
        if vad.is_speech_detected() and not listening:
            listening = True
            print(f"{CLEAR}{DIM}listening...{RESET}", end="", flush=True)
            if live is not None:
                live.start(cut(seen - RATE // 2, seen))
            if incr:                                  # 1 s before the detection: the VAD start can be earlier
                try:
                    sess = stream_incr.IncrSession(whistle.gpu, gpu_fe=gpu_front)
                    # 2026-11-09: the VAD fires late on quiet openings (test clip: speech found at 3.8 s, the old 1 s
                    # preroll dropped the first ~3 s of words). Start from PRE_ROLL before the detection, never
                    # before the previous phrase ended (no duplicate text).
                    pre0 = max(seen - int(PRE_ROLL * RATE), last_end, seen - len(history))
                    sess.feed(to_pcm16(history[pre0 - (seen - len(history)):seen - (seen - len(history))]))
                except Exception as e:
                    log(f"incremental start failed, batch for this phrase: {e}")
                    sess = None
        elif listening and sess is not None:
            try:
                sess.feed(to_pcm16(chunk))
            except Exception as e:
                log(f"incremental feed failed, batch for this phrase: {e}")
                sess = None
        if live is not None:
            if quick is not None and quick.is_speech_detected():
                live.paused = False                   # talking again after a short pause
            live.feed(chunk)
        if quick is not None:
            quick.accept_waveform(chunk)
            while not quick.empty():
                start, n = quick.front.start, len(quick.front.samples)
                quick.pop()
                if p_start is None:                                  # first piece of a new phrase
                    p_start = start
                if live is not None:
                    live.paused = True                # quiet: no streaming pass while the early text runs
                    live.wait_idle()
                ready = (p_start, start + n, run(p_start, start + n, "early"))
        while not vad.empty():
            start, n = vad.front.start, len(vad.front.samples)
            vad.pop()
            listening = False
            last_end = seen
            if live is not None:
                live.end()
            print(flush=True)
            end = start + n
            if ready and abs(ready[1] - end) < RATE // 4 and abs(ready[0] - start) < RATE // 4:
                raw, span, why = ready[2], (ready[0], ready[1]), "early"   # early text covers this phrase
                log("used early text")
            else:
                if live is not None:
                    live.wait_idle()
                raw = None
                if sess is not None:
                    try:
                        raw = whistle.punctuate(sess.finish()[1])   # 2026-11-08: incremental text had no punctuation
                        log(f"incremental: text {sess.tail_s * 1e3:.0f} ms after speech end")
                        span, why = (start, end), "incr"
                    except Exception as e:
                        log(f"incremental finish failed, batch for this phrase: {e}")
                sess = None
                if raw is None:
                    raw, span, why = run(start, end, "full"), (start, end), "full"
            ready, p_start = None, None
            p_stop = kws_says_stop(cut(*span))
            if p_stop is not None:
                log(f"model p(stop) {p_stop:.2f}")
            if p_stop is not None and p_stop >= KWS_STOP and not split_stop(re.sub(r"[^\w\s]", "", raw.lower()).strip())[1]:
                log(f"stop (model {p_stop:.2f}; Whistle heard {raw!r})")
                raw = "stop dictation"
            if not re.search(r"[A-Za-z0-9]", raw or ""):   # 2026-11-06: noise gave "." (44 times in the log): nothing typed
                log(f"no words ({why}) raw={raw!r}: noise, nothing typed")
                continue
            keep_going = handle(raw)
            if args.wav:                 # test runs: speech end (VAD) -> typed, measured
                log(f"delay after speech end {time.time() - t_feed - end / RATE:.2f} s")
            if saving:                                   # after typing, so it never delays the text
                try:
                    save_clip(cut(*span), raw, why)
                except OSError as e:
                    log(f"clip not saved: {e}")
            if not keep_going:
                return
        if held and time.time() - held > STOP_HOLD_SECONDS:
            held = None
            if not handle("Stop", literal=True):
                return

if __name__ == "__main__":
    main()
