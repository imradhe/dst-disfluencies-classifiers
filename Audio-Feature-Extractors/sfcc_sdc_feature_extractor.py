"""
sfcc_sdc_feature_extractor.py
=============================
Single-Frequency-Filtered Cepstral Coefficients (SFFCC) + modified
Shifted Delta Cepstral (SDC) feature extractor.

Reproduces the SFFCC + SDC feature set from:
    Mehrotra et al., "Towards improving Disfluency Detection from Speech
    using Shifted Delta Cepstral Coefficients", IC3 2022.

The filename "sfcc_sdc" is preserved for backwards compatibility with
existing pipelines but the class name is `SFFCCSDCExtractor` and the
algorithm implemented is Single-Frequency Filtering (not sub-band
filterbank cepstral coefficients as an earlier version of this file did).

SFFCC (Section 2.2, equations 2-7)
----------------------------------
For each desired frequency f_k = k * df, k = 1 .. (fs/2)/df:

    s_hat[n] = s[n] * exp(-j*2*pi*n*f_hat_k / fs)     (2)
    where f_hat_k = fs/2 - f_k

    Single-pole IIR:  H(z) = 1 / (1 + a * z^-1)       (3)
    y_k[n] = -a * y_k[n-1] + s_hat[n]                 (4)

    m_k[n] = sqrt(y_r[n]^2 + y_i[n]^2)                (5)
    where y_r, y_i are the real and imaginary parts of y_k[n]

At each 10 ms frame time the SFF spectrum is m_k sampled across k.
The SFFCC feature vector for that frame is:

    S_k[n] = IFFT( log10( m_k[n] ) )                  (7)

with the first N_SFFCC (13 by default) coefficients kept per frame.

SDC
---
Same modified variant used in mfcc+sdc_feature_extractor.py:

    dc(t, i) = c(t + i*p + d) - c(t + i*p - d)

with i in [-K, K] (paper's Section 3.3 modification).
Paper parameters: N = 13 (SFFCC dim), d = 1, p = 2.

Feature composition per segment
-------------------------------
    1. Compute static SFFCC (n_sffcc-dim per frame).
    2. Per-segment mean-variance normalization on SFFCC.
    3. Modified SDC on the MVN'd static SFFCC.
    4. Concatenate static SFFCC + SDC along the feature axis.
    5. Mean-pool across time -> one fixed-size vector per segment
       (segment-level pipeline contract inherited from BaseFeatureExtractor).

Default sizes with n_sffcc = 13, K = 7:
    static  : 13
    SDC     : 13 * (2K + 1) = 195
    total   : 208
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from base_feature_extractor import BaseFeatureExtractor

EPS: float = float(np.finfo(np.float32).eps)


class SFFCCSDCExtractor(BaseFeatureExtractor):
    """
    SFFCC + modified SDC feature extractor (Paper 2, Section 2.2 + 3.3).
    """

    name = "SFFCC_SDC"

    def __init__(
        self,
        sample_rate: int = 16000,
        n_sffcc: int = 13,
        sff_a: float = 0.985,
        sff_df: float = 20.0,
        hop_ms: float = 10.0,
        sdc_d: int = 1,
        sdc_p: int = 2,
        sdc_k: int = 7,
        sdc_mode: str = "modified",
        include_static: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        sample_rate : int
            Audio sampling rate in Hz.
        n_sffcc : int
            Number of static SFFCC coefficients kept per frame (paper: 13).
        sff_a : float
            Single-pole IIR coefficient. Paper: 0.985.
        sff_df : float
            Desired-frequency spacing in Hz. Paper: 20.
        hop_ms : float
            Frame hop for the SFFCC in milliseconds. Paper: 10.
        sdc_d : int
            SDC delta step. Paper: 1.
        sdc_p : int
            Shift between successive SDC blocks. Paper: 2.
        sdc_k : int
            Number of shifts (K in the paper). See `sdc_mode`.
        sdc_mode : str
            "modified"     -> i in [-K, K]  (2K+1 blocks; paper Section 3.3)
            "conventional" -> i in [0, K-1] (K blocks; standard SDC baseline)
        include_static : bool
            If True (default) the final vector is [static | SDC].
            If False the final vector is SDC only.
        """

        super().__init__(sample_rate=sample_rate)

        if sdc_mode not in ("modified", "conventional"):
            raise ValueError(
                f"sdc_mode must be 'modified' or 'conventional', got {sdc_mode!r}"
            )

        self.n_sffcc = n_sffcc
        self.sff_a = sff_a
        self.sff_df = sff_df
        self.hop_length = int(sample_rate * hop_ms / 1000)

        self.sdc_d = sdc_d
        self.sdc_p = sdc_p
        self.sdc_k = sdc_k
        self.sdc_mode = sdc_mode
        self.include_static = include_static

        if self.sdc_mode == "modified":
            self.sdc_blocks = 2 * self.sdc_k + 1
        else:
            self.sdc_blocks = self.sdc_k

        self.sdc_dim = self.n_sffcc * self.sdc_blocks

        self.feature_dim = (
            self.n_sffcc + self.sdc_dim if self.include_static else self.sdc_dim
        )

        # Desired frequency grid f_k = k * df, k = 1 .. (fs/2)/df
        self.n_freqs = int((self.sample_rate / 2.0) / self.sff_df)
        k = np.arange(1, self.n_freqs + 1, dtype=np.float64)
        self._f_k = k * self.sff_df                       # (n_freqs,)
        self._f_hat_k = (self.sample_rate / 2.0) - self._f_k

    # ------------------------------------------------------------
    # SFFCC computation
    # ------------------------------------------------------------

    def _compute_sffcc(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute SFFCC frames for one signal.

        Returns
        -------
        np.ndarray
            Shape: (n_sffcc, n_frames)
        """

        n_samples = int(len(signal))
        if n_samples <= 1:
            return np.zeros((self.n_sffcc, 0), dtype=np.float32)

        # Frame time indices (samples). Centered semantics not needed
        # because SFF is a sample-rate envelope; we simply decimate.
        frame_hops = np.arange(0, n_samples, self.hop_length)
        n_frames = int(len(frame_hops))

        # Cache the sample index array once.
        n = np.arange(n_samples, dtype=np.float64)

        # IIR:  y[n] = -a * y[n-1] + s_hat[n]  =>  b=[1], a=[1, a]
        b = np.array([1.0])
        a_coeffs = np.array([1.0, self.sff_a])

        envelope_matrix = np.empty(
            (self.n_freqs, n_frames), dtype=np.float64
        )

        # Loop over desired frequencies. Each iteration is a full-length
        # complex lfilter (~O(N) in C), fast enough for typical segments.
        signal64 = signal.astype(np.float64, copy=False)

        for k_idx in range(self.n_freqs):
            phase = -2.0 * np.pi * self._f_hat_k[k_idx] * n / self.sample_rate
            s_hat = signal64 * np.exp(1j * phase)

            y = lfilter(b, a_coeffs, s_hat)
            envelope = np.abs(y)

            envelope_matrix[k_idx] = envelope[frame_hops]

        log_env = np.log10(envelope_matrix + EPS)                    # (n_freqs, n_frames)
        cepstrum = np.real(np.fft.ifft(log_env, axis=0))             # (n_freqs, n_frames)
        sffcc = cepstrum[: self.n_sffcc].astype(np.float32)          # (n_sffcc, n_frames)

        return sffcc

    # ------------------------------------------------------------
    # MVN + SDC
    # ------------------------------------------------------------

    @staticmethod
    def _mvn(features: np.ndarray) -> np.ndarray:
        if features.shape[1] == 0:
            return features
        mean = features.mean(axis=1, keepdims=True)
        std = features.std(axis=1, keepdims=True)
        return (features - mean) / (std + EPS)

    def _compute_sdc(self, base_features: np.ndarray) -> np.ndarray:
        """
        SDC per paper equation (1):
            dc(t, i) = c(t + i*p + d) - c(t + i*p - d)

        Shift range depends on `sdc_mode`:
            "modified"     -> i in [-K, K]
            "conventional" -> i in [0, K-1]
        """

        N, T = base_features.shape
        d = self.sdc_d
        p = self.sdc_p
        K = self.sdc_k

        if T == 0:
            return np.zeros((self.sdc_dim, 0), dtype=np.float32)

        if self.sdc_mode == "modified":
            shift_indices = range(-K, K + 1)
            max_offset = K * p + d
        else:
            shift_indices = range(0, K)
            max_offset = (K - 1) * p + d

        padded = np.pad(
            base_features,
            ((0, 0), (max_offset, max_offset)),
            mode="edge",
        )

        shifted_blocks = []
        for i in shift_indices:
            shift = i * p
            plus = padded[:, max_offset + shift + d : max_offset + shift + d + T]
            minus = padded[:, max_offset + shift - d : max_offset + shift - d + T]
            shifted_blocks.append(plus - minus)

        sdc = np.concatenate(shifted_blocks, axis=0).astype(np.float32)

        assert sdc.shape == (self.sdc_dim, T), (
            f"Unexpected SDC shape: {sdc.shape}, expected ({self.sdc_dim}, {T})"
        )

        return sdc

    # ------------------------------------------------------------
    # BaseFeatureExtractor interface
    # ------------------------------------------------------------

    def extract_segment(
        self, audio: np.ndarray, start: float, end: float
    ) -> np.ndarray:
        """
        Extract one fixed-size feature vector for the audio window [start, end].

        Returns
        -------
        np.ndarray
            Shape: (feature_dim,)  = (n_sffcc + sdc_dim,) when include_static
        """

        start_sample = max(0, int(round(start * self.sample_rate)))
        end_sample = max(start_sample, int(round(end * self.sample_rate)))

        segment = audio[start_sample:end_sample]

        if len(segment) < 2:
            return np.zeros(self.feature_dim, dtype=np.float32)

        # 1. Static SFFCC per frame
        static = self._compute_sffcc(segment)

        # 2. Per-segment MVN on the static SFFCC before SDC
        static_mvn = self._mvn(static)

        # 3. Modified SDC on the MVN'd static SFFCC
        sdc = self._compute_sdc(static_mvn)

        # 4. Concatenate along the feature axis
        if self.include_static:
            combined = np.concatenate([static_mvn, sdc], axis=0)
        else:
            combined = sdc

        # 5. Mean-pool across time -> one vector per segment
        vec = combined.mean(axis=1).astype(np.float32)

        assert vec.shape == (self.feature_dim,), (
            f"Unexpected feature shape {vec.shape}, expected ({self.feature_dim},)"
        )

        return vec


# ------------------------------------------------------------------
# Backwards-compatible alias
# ------------------------------------------------------------------

# Some upstream code still imports SFCCSDCExtractor by name.
SFCCSDCExtractor = SFFCCSDCExtractor


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    from pathlib import Path
    from base_feature_extractor import run_extraction

    # Edit these paths for your environment.
    DATASET_DIR = Path(r"D:\IED_Dataset")
    OUTPUT_DIR = Path(r"D:\IED_SFFCC_SDC")

    extractor = SFFCCSDCExtractor(sdc_k=7, sdc_mode="modified")

    run_extraction(
        extractor=extractor,
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        tag=f"sffcc_sdc_{extractor.sdc_mode}_k{extractor.sdc_k}",
    )
