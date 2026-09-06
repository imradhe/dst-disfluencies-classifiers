"""
IED Disfluency Dataset -- Prosody + (optional) Acoustic Feature Extraction
==========================================================================

Reproduces the syllable-level prosody feature vector from:
    Mehrotra et al., "Detecting Multiple Disfluencies from Speech using
    Pre-linguistic Automatic Syllabification with Acoustic and Prosody
    Features", APSIPA-ASC 2021.

Pipeline
--------
1. Pre-linguistic automatic syllabification (Section II):
       WAV -> Gammatone (20 log-spaced filters, 100 Hz to Nyquist) ->
       amplitude envelopes at 1 kHz -> damped harmonic oscillator
       (f0 = 5 Hz, Delta_f = 6 Hz) -> log-sum of the 10 most energetic
       bands -> sonority envelope -> boundary = each local minimum
       preceded by a local maximum.

       Note: the paper caps the gammatone upper edge at 7500 Hz; here it
       is left at Nyquist (8000 Hz for 16 kHz audio). This is a small
       deviation; the log-spaced channels below 7500 Hz are unchanged.

2. Per-syllable 32-dim prosody vector (Table II):

       F0 (15)
           F0 contour     - avg, std, max, min, skew   (5)
           F0 tilt        - avg, std, max, min, skew   (5)
           F0 MSE         - avg, std, max, min, skew   (5)
       Energy (9)
           Energy contour - avg, std, skew             (3)
           Energy tilt    - avg, std, skew             (3)
           Energy MSE     - avg, std, skew             (3)
       Duration (8)
           Pause duration - avg, std, min, max, skew   (5)
           Voiced / unvoiced ratio                     (1)
           Voiced / pause  ratio                       (1)
           Unvoiced / pause ratio                      (1)

       Definitions used here (the paper does not spell them out):
           tilt(t)  = frame-to-frame slope of the contour (numpy diff)
           MSE(t)   = squared residual of the contour vs its global
                      linear fit (a per-frame signal)
           voiced   iff F0(t) > 0
           unvoiced iff F0(t) == 0 and frame energy above pause threshold
           pause    iff frame energy below 10% of segment-max energy

3. Optional +/- 1 syllable stacking (Section IV.A):
       stack current syllable's 32-dim vector with the vectors of its
       predecessor and successor -> 96-dim per syllable.

4. Combining with a BiLSTM-learned acoustic representation (Section IV.B):
       The paper's 90-dim per-syllable acoustic representation comes from
       a BiLSTM trained jointly with the DNN classifier on frame-level
       39-dim MFCC (13 + Delta + Delta-Delta). Because that BiLSTM is a
       trained module rather than a fixed transform, its embeddings are
       produced by the classifier stage, not this extractor. The frame-
       level 39-dim MFCC inputs it consumes can be produced from
       mfcc_feature_extractor.py by setting N_MFCC = 13 and disabling
       the extra energy channel; the classifier pipeline is then
       responsible for grouping frames by syllable, +/- 1 stacking, and
       running the BiLSTM.

Run:
    python ied_prosodic_acoustic_feature_extractor.py
"""

from __future__ import annotations

import csv
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import parselmouth
import soundfile as sf
from scipy.signal import bilinear, find_peaks, hilbert, lfilter, savgol_filter
from scipy.stats import skew
from gammatone.gtgram import gtgram

warnings.filterwarnings("ignore")

# ============================================================================== 
# CONFIGURATION — change paths here, nowhere else
# ============================================================================== 

DATASET_DIR = Path(r"e:\Whisper\IED Dataset")
OUTPUT_DIR = Path(r"e:\Whisper\output")
SAMPLE_RATE = 16_000

# Processing in chunks prevents large recordings from consuming excessive RAM.
CHUNK_DURATION = 30.0
CHUNK_OVERLAP = 0.5

# ============================================================================== 
# LABEL NORMALISATION
# ============================================================================== 

VALID_LABELS = {"I", "PR", "PhR", "WR", "PWR", "P", "Fluent"}

LABEL_MAP: Dict[str, str] = {
    "W.R": "WR", "W.R.": "WR", "WR": "WR", "wr": "WR",
    "PWR": "PWR", "PW.R": "PWR", "P.W.R": "PWR", "PWr": "PWR",
    "PHR": "PhR", "PH.R": "PhR", "Ph.r": "PhR", "Ph.R": "PhR", "PhR": "PhR",
    "PR": "PR", "Pr": "PR", "P.R": "PR", "P.r": "PR",
    "I": "I", "i": "I",
    "P": "P",
    "Fluent": "Fluent",
}

# ============================================================================== 
# DATACLASSES — shared output structure
# ============================================================================== 


@dataclass
class AnnotationRow:
    """One labelled interval from a TXT annotation file."""

    file: str
    start: float
    end: float
    label: str
    duration: float
    raw_label: str
    source_line: int
    has_overlap: bool = False


@dataclass
class FeatureRecord:
    """One fixed-size prosodic/acoustic feature vector."""

    file: str
    start: float
    end: float
    label: str
    duration: float
    has_overlap: bool
    features: np.ndarray


@dataclass
class DatasetStore:
    """Complete extracted dataset ready for classification."""

    X: np.ndarray
    labels: np.ndarray
    files: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    durations: np.ndarray
    has_overlap: np.ndarray
    feature_dim: int
    extractor_name: str


# ============================================================================== 
# AUDIO LOADER
# ============================================================================== 


class AudioLoader:
    """Load WAV files as mono float32 arrays at the required sample rate."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate

    def load(self, wav_path: Path) -> np.ndarray:
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)

        if audio.shape[1] > 1:
            audio = audio.mean(axis=1)
        else:
            audio = audio[:, 0]

        if sr != self.sample_rate:
            raise ValueError(
                f"Expected {self.sample_rate} Hz, got {sr} Hz in {wav_path.name}"
            )

        return audio


# ============================================================================== 
# ANNOTATION PARSER
# ============================================================================== 


class AnnotationParser:
    """Parse TXT annotations and normalise labels."""

    def __init__(
        self,
        valid_labels: set = VALID_LABELS,
        label_map: Dict[str, str] = LABEL_MAP,
    ):
        self.valid_labels = valid_labels
        self.label_map = label_map

    @staticmethod
    def _normalise_label(raw_label: str) -> Optional[str]:
        label = raw_label.strip()
        label = re.sub(r"\([^)]*\)", "", label).strip()

        # Compound annotation such as PR,I is handled by splitting in parse_file.
        mapped = LABEL_MAP.get(label)
        if mapped is not None:
            return mapped

        return label if label in VALID_LABELS else None

    def parse_file(self, txt_path: Path) -> List[AnnotationRow]:
        rows: List[AnnotationRow] = []

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for source_line, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = re.split(r"\s+", line)
                if len(parts) < 3:
                    continue

                try:
                    start = float(parts[0])
                    end = float(parts[1])
                except ValueError:
                    continue

                if end <= start:
                    continue

                raw_label = str(parts[2]).strip()

                # Preserve the established handling of compound labels.
                raw_parts = [p.strip() for p in raw_label.split(",") if p.strip()]
                if not raw_parts:
                    continue

                for raw_part in raw_parts:
                    label = self._normalise_label(raw_part)
                    if label is None:
                        continue

                    rows.append(
                        AnnotationRow(
                            file=txt_path.stem,
                            start=start,
                            end=end,
                            label=label,
                            duration=end - start,
                            raw_label=raw_part,
                            source_line=source_line,
                        )
                    )

        # Mark temporal overlap with another annotation.
        for i, row in enumerate(rows):
            for j, other in enumerate(rows):
                if i == j or row.file != other.file:
                    continue
                if row.start < other.end and other.start < row.end:
                    row.has_overlap = True
                    break

        return rows

    def parse_dataset(self, dataset_dir: Path) -> List[AnnotationRow]:
        rows: List[AnnotationRow] = []
        for txt_path in sorted(dataset_dir.rglob("*.txt")):
            rows.extend(self.parse_file(txt_path))
        return rows


# ============================================================================== 
# BASE FEATURE EXTRACTOR
# ============================================================================== 


class BaseFeatureExtractor:
    """Shared interface for feature extractors."""

    name = "Base"

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.audio_loader = AudioLoader(sample_rate)
        self._failures: List[dict] = []

    def setup(self) -> None:
        """Initialise resources. Prosodic extraction needs no model weights."""
        pass

    def extract_segment(
        self, audio: np.ndarray, start: float, end: float
    ) -> np.ndarray:
        raise NotImplementedError

    def extract_dataset(
        self,
        rows: List[AnnotationRow],
        dataset_dir: Path,
    ) -> DatasetStore:
        """Extract one fixed-size feature vector for every supplied segment."""

        file_order: List[str] = []
        by_file: Dict[str, List[AnnotationRow]] = {}

        for row in rows:
            if row.file not in by_file:
                file_order.append(row.file)
                by_file[row.file] = []
            by_file[row.file].append(row)

        records: List[FeatureRecord] = []
        failures: List[dict] = []
        total_files = len(file_order)
        t_start = time.time()

        for fi, stem in enumerate(file_order, start=1):
            wav_path = self._find_wav(dataset_dir, stem)
            if wav_path is None:
                for row in by_file[stem]:
                    failures.append(
                        dict(
                            file=stem,
                            start=row.start,
                            end=row.end,
                            label=row.label,
                            reason="WAV file not found",
                        )
                    )
                continue

            try:
                audio = self.audio_loader.load(wav_path)
            except Exception as exc:
                for row in by_file[stem]:
                    failures.append(
                        dict(
                            file=stem,
                            start=row.start,
                            end=row.end,
                            label=row.label,
                            reason=f"audio load failed: {exc}",
                        )
                    )
                continue

            ok = 0
            fail = 0
            for row in by_file[stem]:
                try:
                    vec = self.extract_segment(audio, row.start, row.end)
                    vec = np.asarray(vec, dtype=np.float32)

                    if vec.ndim != 1:
                        raise ValueError("Feature vector is not 1-D")
                    if np.isnan(vec).any() or np.isinf(vec).any():
                        raise ValueError("NaN/Inf in feature vector")

                    records.append(
                        FeatureRecord(
                            file=row.file,
                            start=row.start,
                            end=row.end,
                            label=row.label,
                            duration=row.duration,
                            has_overlap=row.has_overlap,
                            features=vec,
                        )
                    )
                    ok += 1
                except Exception as exc:
                    failures.append(
                        dict(
                            file=stem,
                            start=row.start,
                            end=row.end,
                            label=row.label,
                            reason=str(exc),
                        )
                    )
                    fail += 1

            elapsed = time.time() - t_start
            rate = len(records) / elapsed if elapsed > 0 else 0.0
            print(
                f"[{fi:>3}/{total_files}] {stem} "
                f"ok={ok} fail={fail} total={len(records)} rate={rate:.1f} seg/s"
            )

        if not records:
            raise RuntimeError("No features extracted — check dataset path and annotations.")

        store = DatasetStore(
            X=np.stack([r.features for r in records]).astype(np.float32),
            labels=np.array([r.label for r in records]),
            files=np.array([r.file for r in records]),
            starts=np.array([r.start for r in records], dtype=np.float64),
            ends=np.array([r.end for r in records], dtype=np.float64),
            durations=np.array([r.duration for r in records], dtype=np.float32),
            has_overlap=np.array([r.has_overlap for r in records], dtype=bool),
            feature_dim=records[0].features.shape[0],
            extractor_name=self.name,
        )

        self._failures = failures
        return store

    @staticmethod
    def _find_wav(dataset_dir: Path, stem: str) -> Optional[Path]:
        direct = dataset_dir / f"{stem}.wav"
        if direct.exists():
            return direct

        matches = list(dataset_dir.rglob(f"{stem}.wav"))
        return matches[0] if matches else None

    def save(
        self,
        store: DatasetStore,
        npz_path: Path,
        csv_path: Path,
        failures_csv_path: Optional[Path] = None,
    ) -> None:
        """Save standardized NPZ features, metadata CSV, and optional failures CSV."""

        npz_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            npz_path,
            embeddings=store.X,
            labels=store.labels,
            files=store.files,
            starts=store.starts,
            ends=store.ends,
            durations=store.durations,
            has_overlap=store.has_overlap,
        )

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["idx", "file", "start", "end", "label", "duration", "has_overlap"],
            )
            writer.writeheader()
            for i in range(len(store.files)):
                writer.writerow(
                    {
                        "idx": i,
                        "file": store.files[i],
                        "start": store.starts[i],
                        "end": store.ends[i],
                        "label": store.labels[i],
                        "duration": store.durations[i],
                        "has_overlap": store.has_overlap[i],
                    }
                )

        if failures_csv_path is not None and self._failures:
            with open(failures_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["file", "start", "end", "label", "reason"],
                )
                writer.writeheader()
                writer.writerows(self._failures)

        print(f"Saved features  -> {npz_path}")
        print(f"Saved metadata  -> {csv_path}")
        if failures_csv_path is not None:
            print(f"Saved failures  -> {failures_csv_path}")


# ============================================================================== 
# GAMMATONE / SONORITY / SYLLABLE-LIKE SEGMENTATION
# ============================================================================== 


def compute_gammatone(
    audio_signal: np.ndarray,
    sr: int,
    channels: int = 20,
    f_min: int = 100,
    window_time: float = 0.025,
    hop_time: float = 0.001,
) -> np.ndarray:
    return gtgram(audio_signal, sr, window_time, hop_time, channels, f_min)


def compute_sonority(
    gammatone_features: np.ndarray,
    fs_env: int = 1000,
    center_freq: float = 5,
    bandwidth: float = 6,
) -> np.ndarray:
    amplitude = np.sqrt(np.maximum(gammatone_features, 0))

    m = 1.0
    omega0 = 2 * np.pi * center_freq
    k = m * omega0**2
    d = 2 * np.pi * bandwidth * m

    bz, az = bilinear([1.0], [m, d, k], fs=fs_env)

    osc = np.zeros_like(amplitude)
    for ch in range(amplitude.shape[0]):
        osc[ch] = lfilter(bz, az, amplitude[ch])

    osc_amp = np.zeros_like(osc)
    for ch in range(osc.shape[0]):
        osc_amp[ch] = np.abs(hilbert(osc[ch]))

    channel_energy = np.mean(osc_amp**2, axis=1)
    top10 = np.argsort(channel_energy)[-10:]

    sonority = np.sum(np.log10(osc_amp[top10] + 1e-8), axis=0)
    sonority -= sonority.min()

    if sonority.max() > 0:
        sonority /= sonority.max()

    return sonority


def detect_syllable_regions(
    sonority: np.ndarray,
    fs_env: int = 1000,
    min_extremum_distance_ms: float = 50.0,
) -> pd.DataFrame:
    """
    Paper's boundary rule (Section II, end):
        "each local minima, which is preceded by a local maxima,
         is marked as the boundary for a syllable-like chunk."

    The paper does not specify a smoothing step or prominence thresholds,
    so none are applied. A small distance guard is kept so that adjacent
    samples that both qualify as extrema on a piecewise-constant plateau
    are not double-counted; it does not act as a prominence filter.
    """

    envelope = np.clip(sonority, 0.0, 1.0)

    distance = max(1, int(min_extremum_distance_ms / 1000.0 * fs_env))

    peaks, _ = find_peaks(envelope, distance=distance)
    valleys, _ = find_peaks(-envelope, distance=distance)

    # Keep a valley only if there is at least one peak strictly before it.
    boundaries = [v for v in valleys if np.any(peaks < v)]
    boundary_times = np.asarray(boundaries, dtype=np.float64) / fs_env

    rows = []
    for i in range(len(boundary_times) - 1):
        start = float(boundary_times[i])
        end = float(boundary_times[i + 1])
        rows.append(
            {
                "syllable_id": len(rows) + 1,
                "start": start,
                "end": end,
                "duration": end - start,
            }
        )

    return __import__("pandas").DataFrame(
        rows,
        columns=["syllable_id", "start", "end", "duration"],
    )


# ============================================================================== 
# PROSODIC + ACOUSTIC EXTRACTOR
# ============================================================================== 


# ------------------------------------------------------------------
# Helpers for contour statistics (paper Table II semantics)
# ------------------------------------------------------------------


def _safe_skew(values: np.ndarray) -> float:
    """Return 0.0 for series too short for skew rather than raising."""
    if len(values) < 3 or np.std(values) == 0.0:
        return 0.0
    return float(skew(values))


def _stats_5(values: np.ndarray) -> List[float]:
    """avg, std, max, min, skew  (Table II F0 rows and Pause-duration row)."""
    if len(values) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.max(values)),
        float(np.min(values)),
        _safe_skew(values),
    ]


def _stats_3(values: np.ndarray) -> List[float]:
    """avg, std, skew  (Table II Energy rows)."""
    if len(values) == 0:
        return [0.0, 0.0, 0.0]
    return [
        float(np.mean(values)),
        float(np.std(values)),
        _safe_skew(values),
    ]


def _tilt_series(contour: np.ndarray) -> np.ndarray:
    """
    Time-varying tilt of a contour = frame-to-frame slope, np.diff.
    Returns an empty array if the contour has fewer than 2 samples.
    """
    if len(contour) < 2:
        return np.empty(0, dtype=np.float64)
    return np.diff(contour.astype(np.float64))


def _mse_series(contour: np.ndarray) -> np.ndarray:
    """
    Time-varying MSE of a contour = squared residual against the contour's
    global linear fit. Returns an empty array if the contour has fewer
    than 2 samples.
    """
    if len(contour) < 2:
        return np.empty(0, dtype=np.float64)
    x = np.arange(len(contour), dtype=np.float64)
    y = contour.astype(np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    return residual ** 2


class ProsodicAcousticExtractor(BaseFeatureExtractor):
    """
    Extract Mehrotra 2021's 32-dim per-syllable prosody vector
    (Table II). +/- 1 syllable stacking to 96-dim is a post-processing
    step; see `stack_neighbours` below.
    """

    name = "Prosodic32"

    FEATURE_NAMES = [
        # F0 contour (5)
        "f0_avg", "f0_std", "f0_max", "f0_min", "f0_skew",
        # F0 tilt (5)
        "f0_tilt_avg", "f0_tilt_std", "f0_tilt_max", "f0_tilt_min", "f0_tilt_skew",
        # F0 MSE (5)
        "f0_mse_avg", "f0_mse_std", "f0_mse_max", "f0_mse_min", "f0_mse_skew",
        # Energy contour (3)
        "energy_avg", "energy_std", "energy_skew",
        # Energy tilt (3)
        "energy_tilt_avg", "energy_tilt_std", "energy_tilt_skew",
        # Energy MSE (3)
        "energy_mse_avg", "energy_mse_std", "energy_mse_skew",
        # Pause duration (5)
        "pause_dur_avg", "pause_dur_std", "pause_dur_max", "pause_dur_min", "pause_dur_skew",
        # Duration ratios (3)
        "voiced_over_unvoiced",
        "voiced_over_pause",
        "unvoiced_over_pause",
    ]

    # ------------------------------------------------------------
    # F0 (Praat autocorrelation, paper: 10 ms frame / 5 ms shift)
    # ------------------------------------------------------------

    def _pitch(self, audio: np.ndarray) -> "parselmouth.Pitch":
        snd = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)
        # Paper: 10 ms window, 5 ms shift.
        return snd.to_pitch(
            time_step=0.005,
            pitch_floor=75,
            pitch_ceiling=500,
        )

    def _f0_contour(
        self,
        pitch: "parselmouth.Pitch",
        start: float,
        end: float,
    ) -> np.ndarray:
        """F0 contour restricted to the syllable region (voiced frames only)."""
        f0_values = pitch.selected_array["frequency"].copy()
        f0_times = pitch.xs()
        mask = (f0_times >= start) & (f0_times < end) & (f0_values > 0)
        return f0_values[mask]

    def _voicing_frames(
        self,
        pitch: "parselmouth.Pitch",
        start: float,
        end: float,
    ) -> np.ndarray:
        """1 where F0 > 0 (voiced), 0 where F0 == 0 (unvoiced), inside region."""
        f0_values = pitch.selected_array["frequency"]
        f0_times = pitch.xs()
        mask = (f0_times >= start) & (f0_times < end)
        return (f0_values[mask] > 0).astype(np.int8)

    # ------------------------------------------------------------
    # Energy (STFT, paper: 20 ms frame / 10 ms shift)
    # ------------------------------------------------------------

    ENERGY_FRAME = 320    # 20 ms at 16 kHz
    ENERGY_HOP = 160      # 10 ms at 16 kHz

    def _energy_contour(
        self,
        audio: np.ndarray,
        start: float,
        end: float,
    ) -> np.ndarray:
        """Frame-level energy sum inside the syllable region (paper: sum of squared samples per frame)."""
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.ENERGY_FRAME,
            hop_length=self.ENERGY_HOP,
            center=True,
        )[0]
        # Convert RMS to summed-squared energy per frame.
        energy = (rms ** 2) * self.ENERGY_FRAME
        times = np.arange(len(energy)) * self.ENERGY_HOP / self.sample_rate
        mask = (times >= start) & (times < end)
        return energy[mask]

    # ------------------------------------------------------------
    # Duration / pause stats
    # ------------------------------------------------------------

    PAUSE_THRESHOLD_RATIO = 0.10

    def _pause_durations(self, energy: np.ndarray) -> np.ndarray:
        """
        Detect internal pauses: contiguous runs of frames whose energy is
        below `PAUSE_THRESHOLD_RATIO` * peak_energy. Returns each run's
        duration in seconds.
        """
        if len(energy) == 0 or np.max(energy) <= 0:
            return np.empty(0, dtype=np.float64)

        pause_mask = energy < (self.PAUSE_THRESHOLD_RATIO * np.max(energy))
        frame_dur = self.ENERGY_HOP / self.sample_rate

        durations: List[float] = []
        run = 0
        for is_pause in pause_mask:
            if is_pause:
                run += 1
            else:
                if run > 0:
                    durations.append(run * frame_dur)
                run = 0
        if run > 0:
            durations.append(run * frame_dur)

        return np.asarray(durations, dtype=np.float64)

    def _pause_mask(self, energy: np.ndarray) -> np.ndarray:
        if len(energy) == 0 or np.max(energy) <= 0:
            return np.zeros_like(energy, dtype=bool)
        return energy < (self.PAUSE_THRESHOLD_RATIO * np.max(energy))

    # ------------------------------------------------------------
    # Main per-segment vector
    # ------------------------------------------------------------

    def extract_segment(
        self,
        audio: np.ndarray,
        start: float,
        end: float,
    ) -> np.ndarray:
        pitch = self._pitch(audio)

        # F0 contour + derived tilt / MSE series
        f0 = self._f0_contour(pitch, start, end)
        f0_tilt = _tilt_series(f0)
        f0_mse = _mse_series(f0)

        # Energy contour + derived tilt / MSE series
        energy = self._energy_contour(audio, start, end)
        e_tilt = _tilt_series(energy)
        e_mse = _mse_series(energy)

        # Duration group
        pause_durs = self._pause_durations(energy)
        pause_stats = _stats_5(pause_durs)

        voicing = self._voicing_frames(pitch, start, end)
        n_voiced = int((voicing == 1).sum())
        n_unvoiced_all = int((voicing == 0).sum())

        pause_mask = self._pause_mask(energy)
        n_pause = int(pause_mask.sum())
        # Unvoiced-and-not-pause is what "unvoiced" usually means in ratio
        # contexts; approximate here as the raw unvoiced pitch count minus
        # frames overlapping pause. Without frame-rate alignment we treat
        # them as independent counts.
        n_unvoiced = max(0, n_unvoiced_all - n_pause)

        def _ratio(a: int, b: int) -> float:
            return float(a) / float(b) if b > 0 else 0.0

        vector = np.asarray(
            [
                # F0 contour (5)
                *_stats_5(f0),
                # F0 tilt (5)
                *_stats_5(f0_tilt),
                # F0 MSE (5)
                *_stats_5(f0_mse),
                # Energy contour (3)
                *_stats_3(energy),
                # Energy tilt (3)
                *_stats_3(e_tilt),
                # Energy MSE (3)
                *_stats_3(e_mse),
                # Pause duration (5)
                *pause_stats,
                # Ratios (3)
                _ratio(n_voiced, n_unvoiced),
                _ratio(n_voiced, n_pause),
                _ratio(n_unvoiced, n_pause),
            ],
            dtype=np.float32,
        )

        if len(vector) != len(self.FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(self.FEATURE_NAMES)} features, got {len(vector)}"
            )

        return vector


# ------------------------------------------------------------------
# +/- 1 syllable stacking (Section IV.A, 32 -> 96)
# ------------------------------------------------------------------

def stack_neighbours(
    per_syllable_features: np.ndarray,
    per_syllable_files: np.ndarray,
) -> np.ndarray:
    """
    Stack every syllable's D-dim vector with its predecessor's and
    successor's vectors, giving a 3D-dim per-syllable representation.
    Neighbours are taken only within the same file; file boundaries are
    padded by repeating the current syllable's own vector.

    Parameters
    ----------
    per_syllable_features : np.ndarray
        Shape (N, D). Rows must already be ordered by (file, start).
    per_syllable_files : np.ndarray
        Shape (N,). Parallel array of file identifiers.

    Returns
    -------
    np.ndarray
        Shape (N, 3D).
    """
    if per_syllable_features.ndim != 2:
        raise ValueError("per_syllable_features must be 2-D (N, D)")
    N, D = per_syllable_features.shape

    prev_vec = np.empty_like(per_syllable_features)
    next_vec = np.empty_like(per_syllable_features)

    for i in range(N):
        same_prev = i > 0 and per_syllable_files[i - 1] == per_syllable_files[i]
        same_next = i + 1 < N and per_syllable_files[i + 1] == per_syllable_files[i]

        prev_vec[i] = per_syllable_features[i - 1] if same_prev else per_syllable_features[i]
        next_vec[i] = per_syllable_features[i + 1] if same_next else per_syllable_features[i]

    return np.concatenate([prev_vec, per_syllable_features, next_vec], axis=1)


# ============================================================================== 
# REGION LABEL MATCHING
# ============================================================================== 


def assign_annotation_labels(
    regions,
    annotations: List[AnnotationRow],
):
    """Assign the annotation with maximum temporal overlap to each region."""

    output = []

    for _, region in regions.iterrows():
        r_start = float(region["start"])
        r_end = float(region["end"])

        best = None
        best_overlap = 0.0

        for ann in annotations:
            overlap = max(0.0, min(r_end, ann.end) - max(r_start, ann.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best = ann

        if best is None:
            label = "Fluent"
            ann_start = np.nan
            ann_end = np.nan
        else:
            label = best.label
            ann_start = best.start
            ann_end = best.end

        region_duration = max(0.0, r_end - r_start)
        overlap_ratio = best_overlap / region_duration if region_duration > 0 else 0.0

        output.append(
            {
                "syllable_id": int(region["syllable_id"]),
                "label": label,
                "annotation_start": ann_start,
                "annotation_end": ann_end,
                "overlap_duration": best_overlap,
                "overlap_ratio": overlap_ratio,
            }
        )

    return output


# ============================================================================== 
# MAIN PIPELINE
# ============================================================================== 


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  IED PROSODIC + ACOUSTIC FEATURE EXTRACTION")
    print("=" * 70)
    print(f"Dataset : {DATASET_DIR}")
    print(f"Output  : {OUTPUT_DIR}")

    # Step 1 — annotations
    print("\n[Step 1] Parsing annotations ...")
    parser = AnnotationParser()
    annotations = parser.parse_dataset(DATASET_DIR)
    annotations_by_file: Dict[str, List[AnnotationRow]] = {}
    for row in annotations:
        annotations_by_file.setdefault(row.file, []).append(row)
    print(f"Annotation rows: {len(annotations)}")

    # Step 2 — discover WAV/TXT pairs
    wav_paths = {p.stem: p for p in DATASET_DIR.rglob("*.wav")}
    common_ids = sorted(set(wav_paths) & set(annotations_by_file))
    print(f"Matched WAV + TXT files: {len(common_ids)}")

    extractor = ProsodicAcousticExtractor()
    extractor.setup()

    # Step 3 — segmentation + extraction
    print("\n[Step 2] Extracting features ...")
    records: List[FeatureRecord] = []
    failures: List[dict] = []

    for fi, stem in enumerate(common_ids, start=1):
        wav_path = wav_paths[stem]
        try:
            audio = extractor.audio_loader.load(wav_path)
            duration = len(audio) / SAMPLE_RATE
            file_annotations = annotations_by_file[stem]
            file_regions = []

            chunk_start = 0.0
            step = CHUNK_DURATION - CHUNK_OVERLAP

            while chunk_start < duration:
                chunk_end = min(chunk_start + CHUNK_DURATION, duration)
                sample_start = int(chunk_start * SAMPLE_RATE)
                sample_end = int(chunk_end * SAMPLE_RATE)
                audio_chunk = audio[sample_start:sample_end]

                if len(audio_chunk) < int(0.1 * SAMPLE_RATE):
                    break

                g = compute_gammatone(audio_chunk, SAMPLE_RATE)
                s = compute_sonority(g)
                regions = detect_syllable_regions(s)

                if len(regions) == 0:
                    chunk_start += step
                    continue

                regions["start"] += chunk_start
                regions["end"] += chunk_start
                regions["duration"] = regions["end"] - regions["start"]

                core_start = chunk_start if chunk_start == 0 else chunk_start + CHUNK_OVERLAP
                regions = regions[regions["start"] >= core_start].copy()
                if chunk_end < duration:
                    regions = regions[regions["end"] <= chunk_end].copy()

                for _, region in regions.iterrows():
                    r_start = float(region["start"])
                    r_end = float(region["end"])

                    # Extract from the full audio using global timestamps.
                    vec = extractor.extract_segment(audio, r_start, r_end)

                    label_info = assign_annotation_labels(
                        __import__("pandas").DataFrame([region]),
                        file_annotations,
                    )[0]

                    records.append(
                        FeatureRecord(
                            file=stem,
                            start=r_start,
                            end=r_end,
                            label=label_info["label"],
                            duration=r_end - r_start,
                            has_overlap=False,
                            features=vec,
                        )
                    )

                chunk_start += step

            print(f"[{fi:>3}/{len(common_ids)}] {stem}: {sum(r.file == stem for r in records)} regions")

        except Exception as exc:
            print(f"[{fi:>3}/{len(common_ids)}] FAILED {stem}: {exc}")
            failures.append(
                {"file": stem, "start": 0.0, "end": 0.0, "label": "", "reason": repr(exc)}
            )

    if not records:
        raise RuntimeError("No features extracted.")

    # Step 4 — remove exact duplicate regions and assign sequential IDs.
    unique_records: List[FeatureRecord] = []
    seen = set()
    for record in sorted(records, key=lambda r: (r.file, r.start, r.end)):
        key = (record.file, round(record.start, 6), round(record.end, 6))
        if key in seen:
            continue
        seen.add(key)
        unique_records.append(record)

    records = unique_records

    store = DatasetStore(
        X=np.stack([r.features for r in records]).astype(np.float32),
        labels=np.array([r.label for r in records]),
        files=np.array([r.file for r in records]),
        starts=np.array([r.start for r in records], dtype=np.float64),
        ends=np.array([r.end for r in records], dtype=np.float64),
        durations=np.array([r.duration for r in records], dtype=np.float32),
        has_overlap=np.array([r.has_overlap for r in records], dtype=bool),
        feature_dim=records[0].features.shape[0],
        extractor_name=extractor.name,
    )

    # Step 5 — save both the 32-dim per-syllable vector and the
    # 96-dim +/- 1 syllable stacked version (paper Section IV.A).
    print("\n[Step 3] Saving standardized output ...")
    extractor._failures = failures
    extractor.save(
        store,
        npz_path=OUTPUT_DIR / "prosody32_features.npz",
        csv_path=OUTPUT_DIR / "prosody32_metadata.csv",
        failures_csv_path=OUTPUT_DIR / "prosody32_failures.csv",
    )

    stacked = stack_neighbours(store.X, store.files)
    store_stacked = DatasetStore(
        X=stacked.astype(np.float32),
        labels=store.labels,
        files=store.files,
        starts=store.starts,
        ends=store.ends,
        durations=store.durations,
        has_overlap=store.has_overlap,
        feature_dim=stacked.shape[1],
        extractor_name=extractor.name + "_stacked3",
    )
    extractor.save(
        store_stacked,
        npz_path=OUTPUT_DIR / "prosody96_stacked_features.npz",
        csv_path=OUTPUT_DIR / "prosody96_stacked_metadata.csv",
    )

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"Base feature matrix    : {store.X.shape}  (N x 32)")
    print(f"Stacked feature matrix : {store_stacked.X.shape}  (N x 96)")
    print(f"dtype                  : {store.X.dtype}")
    print(f"NaN                    : {np.isnan(store.X).any()}")
    print(f"Inf                    : {np.isinf(store.X).any()}")
    print(f"Files                  : {len(set(store.files.tolist()))}")
    print(f"Failures               : {len(failures)}")
    print("Label distribution:")
    unique, counts = np.unique(store.labels, return_counts=True)
    for label, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  {label:<10} {count:>7}")
    print("=" * 70)


if __name__ == "__main__":
    main()
