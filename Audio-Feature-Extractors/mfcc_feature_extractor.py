"""
mfcc_feature_extractor.py
IED Disfluency Dataset — MFCC Feature Extraction (Paper 1 spec)
================================================================

Reproduces the MFCC feature set from:
    Garg et al., "Towards a Database For Detection of Multiple Speech
    Disfluencies in Indian English", NCC 2021.

Paper spec (Section IV.A):
    - 25 ms window, 10 ms hop (Hamming), 16 kHz
    - 13 cepstral coefficients + 0th cepstral coefficient (C0) + frame energy
      -> 15-dimensional base per frame
    - Delta and delta-delta appended
      -> 45-dimensional feature vector per frame
    - Per-file mean-variance normalization
    - Optional context stacking of +/- CONTEXT_FRAMES neighbours
      (paper reports +/- 3 frames as the best configuration)

Output per file: (D, T) frame-level features saved as compressed .npz,
where D = 45 without stacking, or 45 * (2 * CONTEXT_FRAMES + 1) with it.

Pipeline
--------
1.  Discover .wav files under DATASET_DIR (and check for matching .txt
    annotation files).
2.  Sanity-test on the first file, visualise.
3.  extract_and_save_mfcc() -- extract 45-dim frame features per file
    and save them (+ timestamps + metadata) as a compressed .npz.
4.  Save an extraction log (.csv) and a metadata summary (.csv).
5.  Spot-check one of the saved .npz files.

Usage
-----
    python mfcc_feature_extractor.py
"""

# ============================================================
# SECTION 0 -- CONFIGURATION
# ============================================================

from pathlib import Path

# -- Data paths ------------------------------------------------
DATASET_DIR = Path(r"D:\IED_Dataset")

# MFCC features will be stored here
OUTPUT_DIR = Path(r"D:\IED_MFCC")

FEATURE_DIR = OUTPUT_DIR / "features"
LOG_DIR = OUTPUT_DIR / "logs"

# -- Audio / frame parameters ---------------------------------
TARGET_SR = 16000

# 25 ms Hamming window at 16 kHz
WIN_LENGTH = int(0.025 * TARGET_SR)

# 10 ms hop at 16 kHz
HOP_LENGTH = int(0.010 * TARGET_SR)

N_FFT = 512
N_MELS = 40
WINDOW = "hamming"

# -- MFCC composition (Paper 1: 45-dim per frame) -------------
# librosa's mfcc(n_mfcc=N) returns C0..C(N-1). Paper 1 uses
# "13 cepstral coefficients + the 0th cepstral coefficient",
# which we read as c1..c13 plus C0 -> 14 unique cepstral coeffs.
# Plus energy = 15 base. With delta and delta-delta = 45.
N_MFCC = 14           # C0 .. C13  -> 14 cepstral coefficients
BASE_DIM = N_MFCC + 1  # + energy  -> 15
FEATURE_DIM_STATIC = BASE_DIM * 3  # + delta + delta-delta -> 45

# -- Per-file mean-variance normalization --------------------
APPLY_MVN = True

# -- Context stacking (paper's best config: +/- 3) ------------
# Set to 0 to disable stacking (features stay 45-dim).
# Set to 3 to stack 3 frames before + current + 3 after -> 45 * 7 = 315.
CONTEXT_FRAMES = 0

# ============================================================
# SECTION 1 -- IMPORTS
# ============================================================

import warnings

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# ============================================================
# SECTION 2 -- FEATURE EXTRACTION
# ============================================================

def _log_energy(
    y: np.ndarray,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    """
    Frame-level log-energy: log(sum(x^2)) computed per frame.

    Returns an array of shape (T,) aligned with librosa's centred framing.
    """

    padded = np.pad(y, frame_length // 2, mode="reflect")

    frames = librosa.util.frame(
        padded,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    energy = np.sum(frames ** 2, axis=0)

    return np.log(energy + np.finfo(np.float32).eps)


def _mvn(features: np.ndarray) -> np.ndarray:
    """
    Per-file, per-dimension mean-variance normalization.

    Parameters
    ----------
    features : np.ndarray
        Shape (D, T).

    Returns
    -------
    np.ndarray
        Shape (D, T), zero mean and unit variance along the time axis.
    """

    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True)

    return (features - mean) / (std + np.finfo(np.float32).eps)


def _stack_context(features: np.ndarray, context: int) -> np.ndarray:
    """
    Stack +/- context neighbouring frames along the feature axis.

    Parameters
    ----------
    features : np.ndarray
        Shape (D, T).
    context : int
        Number of frames on each side.

    Returns
    -------
    np.ndarray
        Shape (D * (2 * context + 1), T). Edges are padded by repeating
        the boundary frames.
    """

    if context <= 0:
        return features

    padded = np.pad(
        features,
        ((0, 0), (context, context)),
        mode="edge",
    )

    stacked = np.concatenate(
        [padded[:, i : i + features.shape[1]] for i in range(2 * context + 1)],
        axis=0,
    )

    return stacked


def compute_mfcc_paper1(
    y: np.ndarray,
    sr: int,
) -> np.ndarray:
    """
    Compute the 45-dim (or 45 * (2*CONTEXT_FRAMES + 1)) per-frame MFCC
    feature vector described in Garg et al. 2021.

    Steps:
        1. 14 MFCCs (C0..C13) via librosa (Hamming, 25 ms / 10 ms).
        2. Frame log-energy appended -> 15-dim base per frame.
        3. Per-file mean-variance normalization on the 15-dim base.
        4. Delta and delta-delta on the normalized base -> 45-dim.
        5. Optional context stacking of +/- CONTEXT_FRAMES neighbours.

    Returns
    -------
    np.ndarray
        Shape (D, T), where
            D = 45                    if CONTEXT_FRAMES == 0
            D = 45 * (2 * C + 1)      if CONTEXT_FRAMES == C > 0
    """

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        window=WINDOW,
    )

    log_energy = _log_energy(
        y=y,
        frame_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
    )

    # Align to the shorter of the two, in case of off-by-one padding.
    T = min(mfcc.shape[1], log_energy.shape[0])
    mfcc = mfcc[:, :T]
    log_energy = log_energy[:T]

    base = np.vstack([mfcc, log_energy[np.newaxis, :]])

    if APPLY_MVN:
        base = _mvn(base)

    delta = librosa.feature.delta(base, order=1)
    delta2 = librosa.feature.delta(base, order=2)

    features = np.vstack([base, delta, delta2]).astype(np.float32)

    assert features.shape[0] == FEATURE_DIM_STATIC, (
        f"Expected {FEATURE_DIM_STATIC} static dims, got {features.shape[0]}"
    )

    if CONTEXT_FRAMES > 0:
        features = _stack_context(features, CONTEXT_FRAMES).astype(np.float32)

    return features


def extract_and_save_mfcc(wav_path):
    """
    Extract MFCC features from one WAV file and save them as NPZ.

    Returns:
        status, output_path, number_of_frames, duration, error
    """

    try:
        relative_path = wav_path.relative_to(DATASET_DIR)
        output_path = FEATURE_DIR / relative_path.with_suffix(".npz")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already processed and readable
        if output_path.exists():
            try:
                existing = np.load(output_path, allow_pickle=True)
                n_frames = existing["mfcc"].shape[1]
                duration = float(existing["duration"])
                existing.close()

                return (
                    "skipped",
                    str(output_path),
                    n_frames,
                    duration,
                    "",
                )
            except Exception:
                pass

        y, sr = librosa.load(
            wav_path,
            sr=TARGET_SR,
            mono=True,
        )

        duration = len(y) / sr

        features = compute_mfcc_paper1(y=y, sr=sr)

        frame_times = librosa.frames_to_time(
            np.arange(features.shape[1]),
            sr=sr,
            hop_length=HOP_LENGTH,
        )

        txt_path = wav_path.with_suffix(".txt")

        np.savez_compressed(
            output_path,
            mfcc=features.astype(np.float32),
            frame_times=frame_times.astype(np.float32),
            sample_rate=np.array(sr),
            duration=np.array(duration),
            n_mfcc=np.array(N_MFCC),
            base_dim=np.array(BASE_DIM),
            feature_dim=np.array(features.shape[0]),
            context_frames=np.array(CONTEXT_FRAMES),
            apply_mvn=np.array(APPLY_MVN),
            n_mels=np.array(N_MELS),
            n_fft=np.array(N_FFT),
            hop_length=np.array(HOP_LENGTH),
            win_length=np.array(WIN_LENGTH),
            window=np.array(WINDOW),
            audio_file=np.array(str(wav_path)),
            annotation_file=np.array(
                str(txt_path) if txt_path.exists() else ""
            ),
        )

        return (
            "processed",
            str(output_path),
            features.shape[1],
            duration,
            "",
        )

    except Exception as e:

        return (
            "failed",
            "",
            0,
            0,
            repr(e),
        )


# ============================================================
# SECTION 3 -- MAIN
# ============================================================

def main() -> None:
    print("Libraries loaded successfully.")

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("Dataset :", DATASET_DIR)
    print("Output  :", OUTPUT_DIR)
    print("Features:", FEATURE_DIR)

    wav_files = sorted(DATASET_DIR.rglob("*.wav"))

    print("Total WAV files found:", len(wav_files))

    if len(wav_files) == 0:
        print("ERROR: No WAV files found.")
        return

    print("\nFirst 10 files:")
    for f in wav_files[:10]:
        print(f)

    pairs = []
    missing_txt = []

    for wav_path in wav_files:
        txt_path = wav_path.with_suffix(".txt")

        if txt_path.exists():
            pairs.append((wav_path, txt_path))
        else:
            missing_txt.append(wav_path)

    print("WAV files              :", len(wav_files))
    print("WAV + TXT pairs        :", len(pairs))
    print("WAV files without TXT  :", len(missing_txt))

    if missing_txt:
        print("\nFirst missing TXT files:")
        for f in missing_txt[:20]:
            print(f)

    stacked_dim = FEATURE_DIM_STATIC * (2 * CONTEXT_FRAMES + 1)

    print("MFCC parameters (Paper 1 spec):")
    print("Sample rate      :", TARGET_SR)
    print("Window / hop     :", WIN_LENGTH, "/", HOP_LENGTH, "samples")
    print("N_MFCC           :", N_MFCC, "(C0..C{})".format(N_MFCC - 1))
    print("Base dim         :", BASE_DIM, "(MFCC + energy)")
    print("Static dim       :", FEATURE_DIM_STATIC, "(base + delta + delta-delta)")
    print("Context frames   :", CONTEXT_FRAMES)
    print("Feature dim / T  :", stacked_dim)
    print("MVN              :", APPLY_MVN)

    # -- Sanity test on first file ----------------------------
    test_wav = wav_files[0]

    print("\nTesting:", test_wav)

    y, sr = librosa.load(
        test_wav,
        sr=TARGET_SR,
        mono=True,
    )

    print("Loaded successfully")
    print("Samples      :", len(y))
    print("Sample rate  :", sr)
    print("Duration     :", len(y) / sr, "seconds")

    features = compute_mfcc_paper1(y=y, sr=sr)

    print("Feature shape:", features.shape)
    print("  ({} dims x {} frames)".format(features.shape[0], features.shape[1]))

    plt.figure(figsize=(15, 6))
    librosa.display.specshow(
        features,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
    )
    plt.colorbar()
    plt.title(f"MFCC (Paper 1, {features.shape[0]}-dim) - {test_wav.name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Feature dimension")
    plt.tight_layout()
    plt.show()

    # -- Batch extraction --------------------------------------
    results = []

    print("=" * 70)
    print("STARTING MFCC EXTRACTION")
    print("=" * 70)

    print("Total WAV files:", len(wav_files))
    print("Output folder :", FEATURE_DIR)
    print()

    for wav_path in tqdm(
        wav_files,
        desc="Extracting MFCC",
        unit="file",
    ):
        status, output_path, n_frames, duration, error = extract_and_save_mfcc(wav_path)

        results.append(
            {
                "audio_file": str(wav_path),
                "output_file": output_path,
                "status": status,
                "frames": n_frames,
                "duration_seconds": duration,
                "error": error,
            }
        )

    print("\nExtraction completed.")

    results_df = pd.DataFrame(results)
    log_path = LOG_DIR / "mfcc_extraction_log.csv"
    results_df.to_csv(log_path, index=False)

    print("Log saved to:", log_path)
    print("\nSummary:")
    print(results_df["status"].value_counts())

    metadata = [
        {
            "audio_file": row["audio_file"],
            "mfcc_file": row["output_file"],
            "status": row["status"],
            "frames": row["frames"],
            "duration_seconds": row["duration_seconds"],
        }
        for row in results
    ]

    metadata_df = pd.DataFrame(metadata)
    metadata_path = OUTPUT_DIR / "mfcc_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    print("Metadata saved:", metadata_path)
    print(
        "Total duration:",
        metadata_df["duration_seconds"].sum() / 3600,
        "hours",
    )
    print("Total MFCC frames:", metadata_df["frames"].sum())

    # -- Spot-check a saved file ------------------------------
    saved_files = list(FEATURE_DIR.rglob("*.npz"))
    print("Saved MFCC files:", len(saved_files))

    if saved_files:
        sample_feature = saved_files[0]
        data = np.load(sample_feature, allow_pickle=True)

        print("\nFile:", sample_feature)
        print("MFCC shape :", data["mfcc"].shape)
        print("Frame times:", data["frame_times"].shape)
        print("Duration   :", float(data["duration"]))
        print("Sample rate:", int(data["sample_rate"]))
        print("Feature dim:", int(data["feature_dim"]))
        print("Context    :", int(data["context_frames"]))

        data.close()


if __name__ == "__main__":
    main()
