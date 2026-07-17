"""Adapter for Fortran AMICA 1.7, including multi-model and rejection output."""

import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

from .base import AmicaAdapter


FORTRAN_BINARY = Path("/home/sesma/refs/sccn-amica/amica17_narval")
FORTRAN_WORKDIR = Path("/tmp/amica_parity")


class FortranAdapter(AmicaAdapter):
    def __init__(self, binary_path=None):
        configured = binary_path or os.environ.get("AMICA_FORTRAN_BIN")
        self._binary = Path(configured) if configured else FORTRAN_BINARY
        self._available = self._binary.exists()

    @property
    def name(self) -> str:
        return "fortran-amica-1.7"

    @property
    def available(self) -> bool:
        return self._available

    def run(
        self,
        data,
        params,
        n_iters,
        shared_sphere=None,
        shared_mean=None,
        log_det_sphere=None,
    ):
        if not self._available:
            return None
        n_channels, n_samples = data.shape
        n_components = int(params.get("pcakeep", n_channels))
        n_models = int(params.get("num_models", 1))
        workdir = FORTRAN_WORKDIR / uuid.uuid4().hex[:10]
        workdir.mkdir(parents=True, exist_ok=False)
        outdir = workdir / "out"
        outdir.mkdir()
        try:
            fdt_path = workdir / "data.fdt"
            data.T.astype(np.float32).tofile(fdt_path)
            param_path = workdir / "run.param"
            self._write_param_file(
                param_path,
                fdt_path,
                outdir,
                n_channels,
                n_samples,
                n_components,
                params,
                n_iters,
            )
            environment = os.environ.copy()
            environment["OMP_NUM_THREADS"] = str(params.get("threads", 4))
            environment["OMP_STACKSIZE"] = "512M"
            started = time.perf_counter()
            completed = subprocess.run(
                [str(self._binary), str(param_path)],
                capture_output=True,
                text=True,
                timeout=int(params.get("timeout_seconds", 1200)),
                env=environment,
                check=False,
            )
            elapsed = time.perf_counter() - started
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Fortran AMICA exited {completed.returncode}: "
                    f"{completed.stderr[-1000:]}"
                )
            ll_raw = self._read_float(outdir / "LL", (int(n_iters),))
            ll_history = ll_raw[np.isfinite(ll_raw) & (ll_raw != 0.0)]
            if not ll_history.size:
                raise RuntimeError("no finite likelihood values in Fortran LL output")

            k = int(params["num_mix"])
            w_raw = self._read_float(outdir / "W", (n_components, n_components, n_models))
            w_models = np.transpose(w_raw, (2, 0, 1))
            c_models = self._read_float(outdir / "c", (n_components, n_models)).T
            gm = self._read_float(outdir / "gm", (n_models,))
            comp_list = self._read_int(
                outdir / "comp_list", (n_components, n_models)
            ) - 1
            if np.any(comp_list < 0):
                raise ValueError("invalid zero or negative component index")
            n_global = int(comp_list.max()) + 1
            a_global = self._read_float(outdir / "A", (n_components, n_global))
            global_density = {
                "alpha": self._read_float(outdir / "alpha", (k, n_global)),
                "mu": self._read_float(outdir / "mu", (k, n_global)),
                "beta": self._read_float(outdir / "sbeta", (k, n_global)),
                "rho": self._read_float(outdir / "rho", (k, n_global)),
            }
            a_models = np.stack(
                [a_global[:, comp_list[:, model]] for model in range(n_models)]
            )
            density_models = {
                name: np.stack(
                    [values[:, comp_list[:, model]] for model in range(n_models)]
                )
                for name, values in global_density.items()
            }
            sphere_full = self._read_float(
                outdir / "S", (n_channels, n_channels)
            )
            sphere = sphere_full[:n_components, :]
            mean = self._read_float(outdir / "mean", (n_channels,))

            model_posteriors, sample_mask = self._read_posteriors(
                outdir / "LLt", n_samples, n_models
            )
            single = n_models == 1
            return {
                "W": w_models[0] if single else w_models,
                "A": a_models[0] if single else a_models,
                "alpha": density_models["alpha"][0] if single else density_models["alpha"],
                "mu": density_models["mu"][0] if single else density_models["mu"],
                "beta": density_models["beta"][0] if single else density_models["beta"],
                "rho": density_models["rho"][0] if single else density_models["rho"],
                "c": c_models[0] if single else c_models,
                "ll_history": np.asarray(ll_history),
                "sphere": sphere,
                "sphere_full": sphere_full,
                "mean": mean,
                "log_det_sphere": 0.0,
                "elapsed": elapsed,
                "n_iter": len(ll_history),
                "gm": gm,
                "model_posteriors": model_posteriors,
                "sample_mask": sample_mask if params.get("do_reject") else None,
                "n_rejected": int(np.sum(~sample_mask)),
                "comp_list": comp_list,
            }
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @staticmethod
    def _write_param_file(
        path,
        fdt_path,
        outdir,
        n_channels,
        n_samples,
        n_components,
        params,
        n_iters,
    ):
        path.write_text(
            f"""files {fdt_path}
outdir {outdir}
num_models {int(params.get('num_models', 1))}
num_mix_comps {int(params['num_mix'])}
data_dim {n_channels}
field_dim {n_samples}
num_samples 1
field_blocksize 1
pcakeep {n_components}
max_threads {int(params.get('threads', 4))}
block_size 256
max_iter {int(n_iters)}
writestep 1
dble_data 0
lrate {params['lrate']}
use_grad_norm {1 if params.get('use_grad_norm', False) else 0}
use_min_dll {1 if params.get('use_min_dll', False) else 0}
min_grad_norm 0.000001
min_dll {params['min_dll']}
do_approx_sphere 1
do_reject {1 if params.get('do_reject', False) else 0}
rejsig {params.get('rejsig', 3.0)}
rejstart {int(params.get('rejstart', 2))}
rejint {int(params.get('rejint', 3))}
numrej {int(params.get('numrej', 5))}
do_newton {1 if params.get('do_newton', True) else 0}
newt_start {int(params['newt_start'])}
minlrate 0.00000001
lratefact 0.5
rholrate {params['rholrate']}
rho0 {params['rho0']}
minrho {params['minrho']}
maxrho {params['maxrho']}
rholratefact 0.5
newt_ramp {int(params['newt_ramp'])}
newtrate {params['newtrate']}
max_decs {int(params['max_decs'])}
update_A 1
update_c 1
update_gm 1
update_alpha 1
update_mu 1
update_beta 1
do_rho 1
invsigmax {params['invsigmax']}
invsigmin {params['invsigmin']}
do_mean 1
do_sphere 1
doPCA 1
doscaling {1 if params['doscaling'] else 0}
scalestep 1
fix_init {1 if params.get('fix_init', True) else 0}
share_comps 0
write_LLt 1
""",
            encoding="utf-8",
        )

    @staticmethod
    def _parse_ll(stdout):
        pattern = re.compile(r"LL\s*=\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)")
        return [float(match.group(1)) for line in stdout.splitlines() if (match := pattern.search(line))]

    @staticmethod
    def _read_float(path, shape):
        if not path.exists():
            raise FileNotFoundError(path)
        raw = np.fromfile(str(path), dtype=np.float64)
        expected = int(np.prod(shape))
        if raw.size != expected:
            raise ValueError(f"{path} has {raw.size} values; expected {expected}")
        return raw.reshape(shape, order="F")

    @staticmethod
    def _read_int(path, shape):
        if not path.exists():
            raise FileNotFoundError(path)
        expected = int(np.prod(shape))
        raw = np.fromfile(str(path), dtype=np.int32, count=expected)
        if raw.size != expected:
            raise ValueError(f"{path} has {raw.size} integers; expected {expected}")
        return raw.reshape(shape, order="F")

    @staticmethod
    def _read_posteriors(path, n_samples, n_models):
        if not path.exists():
            raise FileNotFoundError(path)
        raw = np.fromfile(str(path), dtype=np.float64)
        expected = int(n_samples * (n_models + 1))
        if raw.size != expected:
            raise ValueError(f"{path} has {raw.size} values; expected {expected}")
        rows = raw.reshape(n_samples, n_models + 1)
        model_log_joint = rows[:, :n_models]
        sample_mask = ~np.all(model_log_joint == 0.0, axis=1)
        posterior = np.full((n_models, n_samples), np.nan)
        valid = model_log_joint[sample_mask]
        valid = valid - valid.max(axis=1, keepdims=True)
        probabilities = np.exp(valid)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        posterior[:, sample_mask] = probabilities.T
        return posterior, sample_mask
