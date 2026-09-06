"""
MFCC + SDC Feature Extractor  (Paper 2 spec)
IED Disfluency Dataset

Reproduces the MFCC + modified SDC features from:
    Mehrotra et al., "Towards improving Disfluency Detection from Speech
    using Shifted Delta Cepstral Coefficients", IC3 2022.

Feature configuration
---------------------
Sample rate : 16000 Hz
Window      : 25 ms (Hamming)
Hop         : 10 ms
MFCC        : 13 (C0..C12 via librosa)
Energy      : 1 (frame RMS)
Base dim    : 14

Per-file mean-variance normalization is applied to the 14-dim base
BEFORE SDC computation (paper Section 3.3: "The 14-dimensional feature
vector (13 MFCC + energy) obtained per frame is then mean-variance
normalized and used to obtain SDC features.").

SDC (modified):
    Delta at shift i:  dc(t, i) = c(t + i*p + d) - c(t + i*p - d)
    i ranges over -K .. +K  ->  (2K + 1) shifted delta blocks
    Paper's chosen parameters: N=14, d=1, p=2
        K = 7  for filled pause / prolongation  ->  210 SDC dims
        K = 12 for word / part-word repetition  ->  350 SDC dims

This module defaults to K = 7 -> 210 SDC dims, giving 224-dim total.
Pass sdc_k=12 to the constructor for repetition-type disfluencies.
"""

import numpy as np
import librosa

from base_feature_extractor import BaseFeatureExtractor


class MFCCSDCExtractor(BaseFeatureExtractor):
    """
    MFCC + Shifted Delta Cepstral (SDC) feature extractor.

    Inherits the common dataset-processing pipeline from
    BaseFeatureExtractor.

    The extractor-specific work is:
        1. Slice the requested audio segment.
        2. Compute 13 MFCC coefficients.
        3. Compute frame energy.
        4. Combine them into 14-dimensional base features.
        5. Compute SDC features.
        6. Return a fixed 224-dimensional feature vector.
    """

    name = "MFCC_SDC"

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        win_ms: float = 25.0,
        hop_ms: float = 10.0,
        sdc_N: int = 14,
        sdc_d: int = 1,
        sdc_p: int = 2,
        sdc_k: int = 7,
    ):
        """
        Parameters
        ----------
        sample_rate : int
            Audio sampling rate.

        n_mfcc : int
            Number of MFCC coefficients.

        win_ms : float
            Analysis window size in milliseconds.

        hop_ms : float
            Frame hop size in milliseconds.

        sdc_N : int
            Number of dimensions in the base feature vector.

        sdc_d : int
            Delta step.

        sdc_p : int
            Shift between successive SDC blocks.

        sdc_k : int
            Number of shifts on either side of the current frame.
        """

        super().__init__(sample_rate=sample_rate)

        self.n_mfcc = n_mfcc

        self.win_length = int(self.sample_rate * win_ms / 1000)

        self.hop_length = int(self.sample_rate * hop_ms / 1000)

        self.sdc_N = sdc_N
        self.sdc_d = sdc_d
        self.sdc_p = sdc_p
        self.sdc_k = sdc_k

        # 13 MFCC + 1 energy
        self.base_dim = self.n_mfcc + 1

        # With k=7, the implementation uses
        # 2*k + 1 shifted blocks.
        self.sdc_blocks = 2 * self.sdc_k + 1

        self.sdc_dim = self.base_dim * self.sdc_blocks

        self.feature_dim = self.base_dim + self.sdc_dim

    def setup(self) -> None:
        """
        No model needs to be loaded for MFCC + SDC.

        librosa performs the signal-processing operations
        during extract_segment().
        """
        pass

    def _compute_base_features(self, segment: np.ndarray) -> np.ndarray:
        """
        Compute the 14-dimensional base feature sequence, mean-variance
        normalized per segment (paper Section 3.3).

        Returns
        -------
        np.ndarray
            Shape: (14, T)

            13 MFCC coefficients (Hamming window)
            + 1 frame RMS energy
            mean-variance normalized along the time axis.
        """

        mfcc = librosa.feature.mfcc(
            y=segment,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.win_length,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window="hamming",
        )

        energy = librosa.feature.rms(
            y=segment,
            frame_length=self.win_length,
            hop_length=self.hop_length,
        )

        min_frames = min(mfcc.shape[1], energy.shape[1])

        mfcc = mfcc[:, :min_frames]
        energy = energy[:, :min_frames]

        base_features = np.vstack([mfcc, energy]).astype(np.float32)

        # Per-segment mean-variance normalization (paper Section 3.3).
        mean = base_features.mean(axis=1, keepdims=True)
        std = base_features.std(axis=1, keepdims=True)
        base_features = (base_features - mean) / (std + np.finfo(np.float32).eps)

        return base_features.astype(np.float32)

    def _compute_sdc(self, base_features: np.ndarray) -> np.ndarray:
        """
        Compute modified Shifted Delta Cepstral (SDC) features
        exactly as defined in the paper (Section 2.1, equation 1,
        with the modification described in Section 3.3).

        For each frame t and each shift i in [-K, K]:
            dc(t, i) = c(t + i*p + d) - c(t + i*p - d)

        The (2K + 1) shifted delta vectors are stacked along the
        feature axis, giving N * (2K + 1) SDC dimensions per frame.
        Boundary frames are handled by edge padding on the base
        features (equivalent to replicating the first / last frame).

        Parameters
        ----------
        base_features : np.ndarray
            Shape: (N, T)  where N = 14 (13 MFCC + energy)

        Returns
        -------
        np.ndarray
            Shape: (N * (2K + 1), T)
            Default K = 7  ->  (210, T)
        """

        N, T = base_features.shape
        d = self.sdc_d
        p = self.sdc_p
        K = self.sdc_k

        # Pad enough on both sides that every requested offset
        # (t + i*p +/- d) stays in bounds for all t in [0, T).
        max_offset = K * p + d

        padded = np.pad(
            base_features,
            ((0, 0), (max_offset, max_offset)),
            mode="edge",
        )

        shifted_blocks = []

        for i in range(-K, K + 1):
            shift = i * p

            plus = padded[:, max_offset + shift + d : max_offset + shift + d + T]
            minus = padded[:, max_offset + shift - d : max_offset + shift - d + T]

            shifted_blocks.append(plus - minus)

        sdc = np.concatenate(shifted_blocks, axis=0)

        assert sdc.shape == (self.sdc_dim, T), (
            f"Unexpected SDC shape: {sdc.shape}, expected ({self.sdc_dim}, {T})"
        )

        return sdc.astype(np.float32)

    def extract_segment(
        self, audio: np.ndarray, start: float, end: float
    ) -> np.ndarray:
        """
        Extract MFCC + SDC features from one annotated segment.

        Parameters
        ----------
        audio : np.ndarray
            Full audio signal.
            Shape: (n_samples,)

        start : float
            Segment start time in seconds.

        end : float
            Segment end time in seconds.

        Returns
        -------
        np.ndarray
            Shape: (224,)

            14 base MFCC+energy features
            +
            210 SDC features
            =
            224 dimensions
        """

        # ---------------------------------------------------------
        # 1. Slice the audio segment
        # ---------------------------------------------------------

        start_sample = int(start * self.sample_rate)

        end_sample = int(end * self.sample_rate)

        segment = audio[start_sample:end_sample]

        if len(segment) == 0:
            raise ValueError(f"Empty audio segment: {start:.3f}s - {end:.3f}s")

        # ---------------------------------------------------------
        # 2. Compute 14-dimensional base features
        # ---------------------------------------------------------

        base_features = self._compute_base_features(segment)

        # ---------------------------------------------------------
        # 3. Compute 210-dimensional SDC
        # ---------------------------------------------------------

        sdc_features = self._compute_sdc(base_features)

        # ---------------------------------------------------------
        # 4. Combine base + SDC
        # ---------------------------------------------------------

        combined = np.concatenate([base_features, sdc_features], axis=0)

        # combined shape:
        # (224, T)

        # ---------------------------------------------------------
        # 5. Mean-pool over time
        # ---------------------------------------------------------
        #
        # BaseFeatureExtractor requires ONE fixed-size
        # vector per AnnotationRow:
        #
        #     shape = (D,)
        #
        # Therefore we average across frames.
        # ---------------------------------------------------------

        feature_vector = combined.mean(axis=1)

        feature_vector = feature_vector.astype(np.float32)

        # ---------------------------------------------------------
        # 6. Final dimension check
        # ---------------------------------------------------------

        assert feature_vector.shape == (self.feature_dim,), (
            f"Unexpected feature shape: "
            f"{feature_vector.shape}, "
            f"expected ({self.feature_dim},)"
        )

        return feature_vector
