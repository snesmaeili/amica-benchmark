"""Adapter for the manuscript Python/JAX AMICA implementation."""

import time

import numpy as np

from .base import AmicaAdapter


class AmicaPythonAdapter(AmicaAdapter):
    @property
    def name(self) -> str:
        return "amica"

    def run(
        self,
        data,
        params,
        n_iters,
        shared_sphere=None,
        shared_mean=None,
        log_det_sphere=None,
    ):
        try:
            from amica import Amica, AmicaConfig
        except ImportError:  # archived paper capsule
            from amica_python import Amica, AmicaConfig

        cfg = AmicaConfig(
            num_models=int(params.get("num_models", 1)),
            num_mix_comps=int(params["num_mix"]),
            max_iter=int(n_iters),
            pcakeep=params.get("pcakeep"),
            lrate=float(params["lrate"]),
            lratefact=float(params.get("lratefact", 0.5)),
            minlrate=float(params.get("minlrate", 1e-8)),
            newtrate=float(params["newtrate"]),
            newt_start=int(params["newt_start"]),
            newt_ramp=int(params["newt_ramp"]),
            do_newton=bool(params.get("do_newton", True)),
            rho0=float(params["rho0"]),
            minrho=float(params["minrho"]),
            maxrho=float(params["maxrho"]),
            rholrate=float(params["rholrate"]),
            rholratefact=float(params.get("rholratefact", 0.5)),
            invsigmin=float(params["invsigmin"]),
            invsigmax=float(params["invsigmax"]),
            max_decs=int(params["max_decs"]),
            min_dll=float(params["min_dll"]),
            use_min_dll=bool(params.get("use_min_dll", False)),
            do_reject=bool(params.get("do_reject", False)),
            rejsig=float(params.get("rejsig", 3.0)),
            rejstart=int(params.get("rejstart", 2)),
            rejint=int(params.get("rejint", 3)),
            numrej=int(params.get("numrej", 5)),
            doscaling=bool(params["doscaling"]),
            fix_init=bool(params.get("fix_init", True)),
            sphere_type=params.get("sphere_type", "pca"),
            dtype="float64",
        )
        model = Amica(cfg, random_state=int(params.get("seed", 42)))
        fit_kwargs = {}
        if shared_sphere is not None:
            fit_kwargs["init_sphere"] = shared_sphere
        if shared_mean is not None:
            fit_kwargs["init_mean"] = shared_mean
        started = time.perf_counter()
        result = model.fit(data, **fit_kwargs)
        elapsed = time.perf_counter() - started
        return {
            "W": np.asarray(result.unmixing_matrix_white_),
            "A": np.asarray(result.mixing_matrix_white_),
            "alpha": np.asarray(result.alpha_),
            "mu": np.asarray(result.mu_),
            "beta": np.asarray(result.sbeta_),
            "rho": np.asarray(result.rho_),
            "c": np.asarray(result.c_),
            "ll_history": np.asarray(result.log_likelihood),
            "sphere": np.asarray(result.whitener_),
            "mean": np.asarray(result.mean_),
            "log_det_sphere": float(getattr(model, "log_det_sphere", 0.0)),
            "elapsed": elapsed,
            "n_iter": int(result.n_iter),
            "gm": np.atleast_1d(np.asarray(result.gm_)),
            "model_posteriors": (
                None
                if getattr(result, "model_posteriors_", None) is None
                else np.asarray(result.model_posteriors_)
            ),
            "sample_mask": (
                None
                if getattr(result, "sample_mask_", None) is None
                else np.asarray(result.sample_mask_, dtype=bool)
            ),
            "n_rejected": int(getattr(result, "n_rejected_", 0)),
        }
