"""Pairwise mutual-information matrices for an illustrative subject (supplementary S2).

Loads the committed AMICA fit (_ica.fif), re-loads + preprocesses the SAME subject through the
benchmark runner (load_data -> apply_analysis_window(250) -> preprocess(filter 1-100 Hz + site
notch)) so ica.get_sources is valid, then computes the pairwise MI matrix for the AMICA component
activations and for the input scalp channels (contrast: input has structured off-diagonal MI,
AMICA components collapse toward the diagonal). CPU-only, single subject. Saves an npz.
"""
import argparse

import mne
import numpy as np

from amica_python.benchmark import runner
from amica_python.benchmark.metrics import pairwise_mi_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fif", required=True, help="committed AMICA _ica.fif")
    ap.add_argument("--dataset", default="ds004505")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--max-samples", type=int, default=200000,
                    help="subsample columns for the MI histograms (speed)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    raw, meta = runner.load_data(a.dataset, a.subject, return_metadata=True)
    ica = mne.preprocessing.read_ica(a.fif)
    # Match the exact channels + preprocessing the fit used, without depending on the
    # runner.preprocess() signature (it varies across package versions).
    raw.load_data()
    keep = [c for c in ica.ch_names if c in raw.ch_names]
    raw.pick(keep)                               # exactly the channels the ICA was fit on
    if abs(float(raw.info["sfreq"]) - 250.0) > 1e-6:
        raw.resample(250.0)
    raw.filter(1.0, 100.0, method="fir", fir_design="firwin")
    nf = {"ds004505": 60.0, "ds004504": 50.0, "ds004621": 50.0}.get(a.dataset, 60.0)
    raw.notch_filter(nf, method="fir", fir_design="firwin")

    Y = ica.get_sources(raw).get_data()          # AMICA component activations (C x T)
    X = raw.get_data()                           # input scalp channels (Nch x T)
    if a.max_samples and Y.shape[1] > a.max_samples:
        idx = np.linspace(0, Y.shape[1] - 1, a.max_samples).astype(int)
        Y, X = Y[:, idx], X[:, idx]

    mi_amica = pairwise_mi_matrix(Y)
    mi_input = pairwise_mi_matrix(X)
    np.savez(a.out, mi_amica=mi_amica, mi_input=mi_input,
             n_comp=Y.shape[0], n_ch=X.shape[0],
             dataset=a.dataset, subject=a.subject, n_samples=Y.shape[1])
    od = float(mi_amica[~np.eye(mi_amica.shape[0], dtype=bool)].mean())
    oi = float(mi_input[~np.eye(mi_input.shape[0], dtype=bool)].mean())
    print(f"saved {a.out}: mi_amica {mi_amica.shape} mean-offdiag {od:.4f} bits; "
          f"mi_input {mi_input.shape} mean-offdiag {oi:.4f} bits")


if __name__ == "__main__":
    main()
