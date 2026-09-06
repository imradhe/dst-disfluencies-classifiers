"""
classifier.features
===================
File-level, frame-aligned feature extraction bridge.

Every feature configuration exposes:

    extract_file(audio: np.ndarray, n_target_frames: int | None) -> (D, T)

The T dimension is guaranteed to equal `n_target_frames` when it is
supplied (frames are trimmed or edge-padded to align exactly with the
label timeline built by `classifier.data.frame_times_for`).

Feature configs implemented here:

    mfcc45              -- Extractor #1, Paper 1's 45-dim per frame
    mfcc45_ctx3         -- #1 with ±3 frame stacking -> 315
    mfcc_sdc_mod_k7     -- #2, MFCC + modified SDC, K=7  -> 224
    mfcc_sdc_mod_k12    -- #2, MFCC + modified SDC, K=12 -> 364
    mfcc_sdc_conv_k7    -- #2, MFCC + conventional SDC, K=7 -> 112
    sffcc_sdc_mod_k7    -- #3, SFFCC + modified SDC, K=7 -> 208
    prosody32           -- #4, per-syllable 32-dim, upsampled to per-frame
    prosody96           -- #4, per-syllable ±1 stacked 96-dim, upsampled
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# The paper-faithful extractors live in a sibling directory.
EXTRACTORS_DIR = Path(__file__).resolve().parents[1] / "Audio-Feature-Extractors"
if str(EXTRACTORS_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACTORS_DIR))

from classifier.config import HOP_LENGTH, SAMPLE_RATE


# ---------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------

def _align_to_target(features: np.ndarray, n_target: Optional[int]) -> np.ndarray:
    """
    Trim or right-pad (edge-mode) so features.shape[1] == n_target.
    """
    if n_target is None:
        return features
    D, T = features.shape
    if T == n_target:
        return features
    if T > n_target:
        return features[:, :n_target]
    pad = n_target - T
    return np.pad(features, ((0, 0), (0, pad)), mode="edge")


def _context_stack(features: np.ndarray, context: int) -> np.ndarray:
    """Stack ±context neighbours along the feature axis. features: (D, T)."""
    if context <= 0:
        return features
    padded = np.pad(features, ((0, 0), (context, context)), mode="edge")
    return np.concatenate(
        [padded[:, i:i + features.shape[1]] for i in range(2 * context + 1)],
        axis=0,
    )


# ---------------------------------------------------------------------
# Extractor 1 -- MFCC (45-dim, Paper 1)
# ---------------------------------------------------------------------

def _mfcc45(audio: np.ndarray, n_target: Optional[int],
            context: int = 0) -> np.ndarray:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mfcc_paper1", str(EXTRACTORS_DIR / "mfcc_feature_extractor.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("mfcc_paper1", module)
    spec.loader.exec_module(module)

    # Use the standalone paper-1 helper. Context is applied here so the
    # config knob is per-run, not global.
    saved_ctx = module.CONTEXT_FRAMES
    module.CONTEXT_FRAMES = 0
    try:
        base = module.compute_mfcc_paper1(y=audio, sr=SAMPLE_RATE)   # (45, T)
    finally:
        module.CONTEXT_FRAMES = saved_ctx

    if context > 0:
        base = _context_stack(base, context)
    return _align_to_target(base.astype(np.float32), n_target)


# ---------------------------------------------------------------------
# Extractor 2 -- MFCC + SDC  (frame-level access to the internal (D, T))
# ---------------------------------------------------------------------

_MFCC_SDC_MOD_CLS = None

def _get_mfcc_sdc_cls():
    global _MFCC_SDC_MOD_CLS
    if _MFCC_SDC_MOD_CLS is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mfcc_sdc", str(EXTRACTORS_DIR / "mfcc+sdc_feature_extractor.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("mfcc_sdc", module)
        spec.loader.exec_module(module)
        _MFCC_SDC_MOD_CLS = module.MFCCSDCExtractor
    return _MFCC_SDC_MOD_CLS


def _mfcc_sdc(audio: np.ndarray, n_target: Optional[int],
              sdc_k: int, sdc_mode: str) -> np.ndarray:
    cls = _get_mfcc_sdc_cls()
    ext = cls(sdc_k=sdc_k, sdc_mode=sdc_mode)
    base = ext._compute_base_features(audio)          # (14, T)
    sdc  = ext._compute_sdc(base)                      # (sdc_dim, T)
    combined = np.concatenate([base, sdc], axis=0)    # (feature_dim, T)
    return _align_to_target(combined.astype(np.float32), n_target)


# ---------------------------------------------------------------------
# Extractor 3 -- SFFCC + SDC  (frame-level)
# ---------------------------------------------------------------------

_SFFCC_SDC_CLS = None

def _get_sffcc_sdc_cls():
    global _SFFCC_SDC_CLS
    if _SFFCC_SDC_CLS is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "sffcc_sdc", str(EXTRACTORS_DIR / "sfcc_sdc_feature_extractor.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("sffcc_sdc", module)
        spec.loader.exec_module(module)
        _SFFCC_SDC_CLS = module.SFFCCSDCExtractor
    return _SFFCC_SDC_CLS


def _sffcc_sdc(audio: np.ndarray, n_target: Optional[int],
               sdc_k: int, sdc_mode: str) -> np.ndarray:
    cls = _get_sffcc_sdc_cls()
    ext = cls(sdc_k=sdc_k, sdc_mode=sdc_mode)
    static = ext._compute_sffcc(audio)                # (13, T)
    static_mvn = ext._mvn(static)
    sdc = ext._compute_sdc(static_mvn)                # (sdc_dim, T)
    combined = np.concatenate([static_mvn, sdc], axis=0)
    return _align_to_target(combined.astype(np.float32), n_target)


# ---------------------------------------------------------------------
# Extractor 4 -- Prosody 32 / 96, upsampled per-frame
# ---------------------------------------------------------------------

_PROSODY_MODULE = None

def _get_prosody_module():
    """Load the prosody extractor module, registering in sys.modules so
    dataclass introspection on Python 3.14 finds it."""
    global _PROSODY_MODULE
    if _PROSODY_MODULE is None:
        import importlib.util
        name = "ied_prosodic_acoustic_feature_extractor"
        spec = importlib.util.spec_from_file_location(
            name, str(EXTRACTORS_DIR / f"{name}.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _PROSODY_MODULE = module
    return _PROSODY_MODULE


def _syllabify_chunked(audio: np.ndarray, module,
                       chunk_sec: float = 30.0,
                       overlap_sec: float = 0.5
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Syllabify a long file by chunking through Gammatone -> sonority ->
    detect_syllable_regions per chunk. Returns two 1-D arrays of
    syllable start/end times in seconds (global to the file).

    Matches the chunking pattern used in the extractor's own main().
    Chunking is what makes this affordable on 30-min lecture files.
    """
    duration = len(audio) / SAMPLE_RATE
    step = chunk_sec - overlap_sec
    starts, ends = [], []

    chunk_start = 0.0
    while chunk_start < duration:
        chunk_end = min(chunk_start + chunk_sec, duration)
        s = int(chunk_start * SAMPLE_RATE)
        e = int(chunk_end   * SAMPLE_RATE)
        piece = audio[s:e]
        if len(piece) < int(0.1 * SAMPLE_RATE):
            break

        g = module.compute_gammatone(piece, SAMPLE_RATE)
        son = module.compute_sonority(g)
        regs = module.detect_syllable_regions(son)

        if len(regs):
            # Discard boundary-overlapping regions to avoid duplicates
            core_start = chunk_start if chunk_start == 0 else chunk_start + overlap_sec
            for _, row in regs.iterrows():
                rs = chunk_start + float(row["start"])
                re_ = chunk_start + float(row["end"])
                if rs < core_start:
                    continue
                if chunk_end < duration and re_ > chunk_end:
                    continue
                starts.append(rs)
                ends.append(re_)

        chunk_start += step

    if not starts:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    # Sort and dedupe by (start, end)
    idx = np.lexsort((np.asarray(ends), np.asarray(starts)))
    starts = np.asarray(starts, dtype=np.float64)[idx]
    ends   = np.asarray(ends,   dtype=np.float64)[idx]
    keep = np.ones(len(starts), dtype=bool)
    for i in range(1, len(starts)):
        if abs(starts[i] - starts[i - 1]) < 1e-6 and abs(ends[i] - ends[i - 1]) < 1e-6:
            keep[i] = False
    return starts[keep], ends[keep]


def _prosody_syllable_vectors(audio: np.ndarray,
                              module) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast path: compute pitch + energy contours ONCE per file, syllabify
    in chunks, then per-syllable slice + stat computation.

    Returns
    -------
    regions : np.ndarray of shape (N_syl, 2)
        [start_sec, end_sec] per syllable, sorted by start.
    vectors : np.ndarray of shape (N_syl, 32)
    """
    import parselmouth
    import librosa

    ext = module.ProsodicAcousticExtractor()
    ext.setup()

    # Syllabify (chunked -> affordable on 30-min files).
    starts, ends = _syllabify_chunked(audio, module)
    if len(starts) == 0:
        return (np.zeros((0, 2), dtype=np.float64),
                np.zeros((0, len(ext.FEATURE_NAMES)), dtype=np.float32))

    # ---- Compute pitch once (parselmouth autocorrelation) ----
    snd = parselmouth.Sound(audio, sampling_frequency=SAMPLE_RATE)
    pitch = snd.to_pitch(time_step=0.005, pitch_floor=75, pitch_ceiling=500)
    f0_values = pitch.selected_array["frequency"]
    f0_times  = pitch.xs()

    # ---- Compute energy contour once (RMS -> summed-squared energy) ----
    rms = librosa.feature.rms(
        y=audio,
        frame_length=ext.ENERGY_FRAME,
        hop_length=ext.ENERGY_HOP,
        center=True,
    )[0]
    energy = (rms ** 2) * ext.ENERGY_FRAME
    energy_times = np.arange(len(energy)) * ext.ENERGY_HOP / SAMPLE_RATE

    # ---- Per-syllable stats using pre-computed contours ----
    from ied_prosodic_acoustic_feature_extractor import (
        _mse_series, _stats_3, _stats_5, _tilt_series,
    )

    def _stats_for(start: float, end: float) -> np.ndarray:
        # F0 slice (voiced only)
        mask_f = (f0_times >= start) & (f0_times < end) & (f0_values > 0)
        f0 = f0_values[mask_f]
        # Voicing counts
        mask_all = (f0_times >= start) & (f0_times < end)
        voicing = (f0_values[mask_all] > 0).astype(np.int8)
        n_voiced = int((voicing == 1).sum())
        n_unvoiced_all = int((voicing == 0).sum())

        # Energy slice + pause detection
        mask_e = (energy_times >= start) & (energy_times < end)
        e = energy[mask_e]

        if len(e) and np.max(e) > 0:
            pause_mask = e < (ext.PAUSE_THRESHOLD_RATIO * np.max(e))
        else:
            pause_mask = np.zeros_like(e, dtype=bool)

        # Pause durations (contiguous runs)
        frame_dur = ext.ENERGY_HOP / SAMPLE_RATE
        pause_durs = []
        run = 0
        for is_p in pause_mask:
            if is_p:
                run += 1
            else:
                if run > 0:
                    pause_durs.append(run * frame_dur)
                run = 0
        if run > 0:
            pause_durs.append(run * frame_dur)
        pause_durs = np.asarray(pause_durs, dtype=np.float64)

        n_pause = int(pause_mask.sum())
        n_unvoiced = max(0, n_unvoiced_all - n_pause)

        def _ratio(a, b):
            return float(a) / float(b) if b > 0 else 0.0

        vec = np.asarray(
            [
                *_stats_5(f0),
                *_stats_5(_tilt_series(f0)),
                *_stats_5(_mse_series(f0)),
                *_stats_3(e),
                *_stats_3(_tilt_series(e)),
                *_stats_3(_mse_series(e)),
                *_stats_5(pause_durs),
                _ratio(n_voiced, n_unvoiced),
                _ratio(n_voiced, n_pause),
                _ratio(n_unvoiced, n_pause),
            ],
            dtype=np.float32,
        )
        return vec

    vecs = np.stack([_stats_for(s, e) for s, e in zip(starts, ends)]).astype(np.float32)
    regions = np.column_stack([starts, ends]).astype(np.float64)
    return regions, vecs


def _prosody(audio: np.ndarray, n_target: Optional[int],
             stack_neighbours: bool) -> np.ndarray:
    module = _get_prosody_module()
    regions, vecs = _prosody_syllable_vectors(audio, module)

    if stack_neighbours and len(vecs) > 0:
        # Everything comes from a single file, so the file mask is trivial.
        files = np.zeros(len(vecs), dtype=np.int32)
        vecs = module.stack_neighbours(vecs, files).astype(np.float32)

    D = vecs.shape[1] if len(vecs) else (96 if stack_neighbours else 32)

    n_frames = n_target if n_target is not None else int(
        np.ceil(len(audio) / HOP_LENGTH))
    frame_times = np.arange(n_frames) * HOP_LENGTH / SAMPLE_RATE

    out = np.zeros((D, n_frames), dtype=np.float32)
    if len(vecs) == 0:
        return out

    starts = regions[:, 0]
    ends = regions[:, 1]

    # For each syllable, assign its vector to the frames whose centre
    # falls inside [start, end). Frames outside every region stay 0.
    for i in range(len(vecs)):
        lo = int(np.searchsorted(frame_times, starts[i], side="left"))
        hi = int(np.searchsorted(frame_times, ends[i], side="left"))
        if lo < hi:
            out[:, lo:hi] = vecs[i][:, None]

    return _align_to_target(out, n_target)


# ---------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------

FeatureFn = Callable[[np.ndarray, Optional[int]], np.ndarray]


def _reg() -> Dict[str, Tuple[FeatureFn, int]]:
    """(fn, feature_dim) per feature tag."""
    return {
        "mfcc45":             (lambda a, n: _mfcc45(a, n, context=0), 45),
        "mfcc45_ctx3":        (lambda a, n: _mfcc45(a, n, context=3), 315),
        "mfcc_sdc_mod_k7":    (lambda a, n: _mfcc_sdc(a, n, 7,  "modified"),     224),
        "mfcc_sdc_mod_k12":   (lambda a, n: _mfcc_sdc(a, n, 12, "modified"),     364),
        "mfcc_sdc_conv_k7":   (lambda a, n: _mfcc_sdc(a, n, 7,  "conventional"), 112),
        "sffcc_sdc_mod_k7":   (lambda a, n: _sffcc_sdc(a, n, 7, "modified"),     208),
        "prosody32":          (lambda a, n: _prosody(a, n, stack_neighbours=False), 32),
        "prosody96":          (lambda a, n: _prosody(a, n, stack_neighbours=True),  96),
    }


FEATURE_REGISTRY = _reg()


def extract_file(tag: str, audio: np.ndarray,
                 n_target_frames: Optional[int]) -> np.ndarray:
    """
    Compute frame-level features for one file under a named config.

    Returns
    -------
    np.ndarray of shape (D, n_target_frames or T_extracted).
    """
    if tag not in FEATURE_REGISTRY:
        raise KeyError(f"Unknown feature tag {tag!r}. "
                       f"Known: {sorted(FEATURE_REGISTRY)}")
    fn, _dim = FEATURE_REGISTRY[tag]
    return fn(audio, n_target_frames)


def feature_dim(tag: str) -> int:
    return FEATURE_REGISTRY[tag][1]


def all_tags() -> List[str]:
    return list(FEATURE_REGISTRY.keys())
