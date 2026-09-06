"""
base_feature_extractor.py
=========================
Shared base class + dataset-processing pipeline used by the segment-level
feature extractors (currently mfcc+sdc, sfcc_sdc / sffcc_sdc).

Design mirrors the inline BaseFeatureExtractor in
ied_prosodic_acoustic_feature_extractor.py so the same annotations,
labels, DatasetStore and save layout are used across all extractors.

A subclass must:
    - set the class attribute `name` (str)
    - implement `extract_segment(audio, start, end) -> np.ndarray` returning
      a fixed-size 1-D feature vector for one annotated segment
    - optionally override `setup()` to load models / buffers
"""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf


# ------------------------------------------------------------------
# Label vocabulary and normalisation (kept identical to #4 so that
# the same annotation files parse the same way across extractors)
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Data containers
# ------------------------------------------------------------------

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
    """One fixed-size feature vector for one segment."""

    file: str
    start: float
    end: float
    label: str
    duration: float
    has_overlap: bool
    features: np.ndarray


@dataclass
class DatasetStore:
    """A whole extracted dataset ready for classification."""

    X: np.ndarray
    labels: np.ndarray
    files: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    durations: np.ndarray
    has_overlap: np.ndarray
    feature_dim: int
    extractor_name: str


# ------------------------------------------------------------------
# Audio + annotation helpers
# ------------------------------------------------------------------

class AudioLoader:
    """Load WAV files as mono float32 arrays at the required sample rate."""

    def __init__(self, sample_rate: int = 16000):
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


class AnnotationParser:
    """Parse whitespace-separated `start end label` TXT annotations."""

    def __init__(
        self,
        valid_labels: set = VALID_LABELS,
        label_map: Dict[str, str] = LABEL_MAP,
    ):
        self.valid_labels = valid_labels
        self.label_map = label_map

    def _normalise_label(self, raw_label: str) -> Optional[str]:
        label = raw_label.strip()
        label = re.sub(r"\([^)]*\)", "", label).strip()

        mapped = self.label_map.get(label)
        if mapped is not None:
            return mapped

        return label if label in self.valid_labels else None

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

        # Flag temporally-overlapping annotations within the same file.
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


# ------------------------------------------------------------------
# Shared base class
# ------------------------------------------------------------------

class BaseFeatureExtractor:
    """
    Shared interface for segment-level feature extractors.

    Subclasses:
        - set class attribute `name`
        - implement `extract_segment(audio, start, end) -> np.ndarray`
        - optionally override `setup()`
    """

    name = "Base"

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.audio_loader = AudioLoader(sample_rate)
        self._failures: List[dict] = []

    def setup(self) -> None:
        """Load any models / precomputed resources. Default is no-op."""

    def extract_segment(
        self, audio: np.ndarray, start: float, end: float
    ) -> np.ndarray:
        raise NotImplementedError

    def extract_dataset(
        self,
        rows: List[AnnotationRow],
        dataset_dir: Path,
    ) -> DatasetStore:
        """Extract one fixed-size feature vector per annotated segment."""

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
            raise RuntimeError(
                "No features extracted -- check dataset path and annotations."
            )

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
                fieldnames=[
                    "idx", "file", "start", "end", "label",
                    "duration", "has_overlap",
                ],
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


# ------------------------------------------------------------------
# Simple driver — usable from any subclass's main()
# ------------------------------------------------------------------

def run_extraction(
    extractor: BaseFeatureExtractor,
    dataset_dir: Path,
    output_dir: Path,
    tag: str,
) -> DatasetStore:
    """
    Standard main-loop: parse annotations under `dataset_dir`, extract
    features with `extractor`, save NPZ + metadata + failures under
    `output_dir` using `tag` as the file basename.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  {extractor.name} FEATURE EXTRACTION")
    print("=" * 70)
    print(f"Dataset : {dataset_dir}")
    print(f"Output  : {output_dir}")

    parser = AnnotationParser()
    annotations = parser.parse_dataset(dataset_dir)
    print(f"Annotation rows: {len(annotations)}")

    extractor.setup()
    store = extractor.extract_dataset(annotations, dataset_dir)

    extractor.save(
        store,
        npz_path=output_dir / f"{tag}_features.npz",
        csv_path=output_dir / f"{tag}_metadata.csv",
        failures_csv_path=output_dir / f"{tag}_failures.csv",
    )

    print("=" * 70)
    print(f"Feature matrix : {store.X.shape}")
    print(f"Feature dim    : {store.feature_dim}")
    print(f"Files          : {len(set(store.files.tolist()))}")
    print(f"Failures       : {len(extractor._failures)}")
    print("=" * 70)

    return store
