"""Adapter for Fortran AMICA binary (amica15ub — statically linked, no MPI)."""

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .base import AmicaAdapter

# amica15ub is statically linked (no MPI needed), works on any node
FORTRAN_BINARY = Path("/home/sesma/refs/sccn-amica/amica15ub")
# Use a short fixed path to avoid Fortran buffer overflow on long tmpdir paths
FORTRAN_WORKDIR = Path("/tmp/amica_parity")


class FortranAdapter(AmicaAdapter):

    def __init__(self, binary_path=None):
        self._binary = Path(binary_path) if binary_path else FORTRAN_BINARY
        self._available = self._binary.exists()

    @property
    def name(self) -> str:
        return "fortran"

    @property
    def available(self) -> bool:
        return self._available

    def run(self, data, params, n_iters, shared_sphere=None,
            shared_mean=None, log_det_sphere=None):
        if not self._available:
            return None

        n_ch, n_samples = data.shape
        n_comp = params.get("pcakeep", n_ch)

        # Use short fixed path to avoid Fortran char buffer overflow
        workdir = FORTRAN_WORKDIR
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True)
        outdir = workdir / "out"
        outdir.mkdir()

        try:
            # Write FDT file (float32, Fortran column-major)
            fdt_path = workdir / "data.fdt"
            data.T.astype(np.float32).tofile(fdt_path)

            # Write param file
            param_path = workdir / "run.param"
            self._write_param_file(
                param_path, fdt_path, outdir,
                n_ch, n_samples, n_comp, params, n_iters,
            )

            # Run Fortran binary
            env = {**os.environ}
            env["OMP_NUM_THREADS"] = str(params.get("omp_threads", 4))
            env["OMP_STACKSIZE"] = "512M"

            t0 = time.perf_counter()
            try:
                result = subprocess.run(
                    [str(self._binary), str(param_path)],
                    capture_output=True, text=True, timeout=600,
                    env=env,
                )
                elapsed = time.perf_counter() - t0
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
                print(f"Fortran execution error: {e}")
                return None

            if result.returncode != 0:
                # Show first meaningful error line
                err = result.stderr.strip().splitlines()
                msg = next((l for l in err if "error" in l.lower()
                            or "terminate" in l.lower()), err[0] if err else "unknown")
                print(f"Fortran exit code {result.returncode}: {msg}")
                return None

            # Parse LL trajectory from stdout
            ll_history = self._parse_ll(result.stdout)
            if not ll_history:
                print("Fortran: no LL values parsed from stdout")
                return None

            # Read output files
            J = params["num_mix"]
            W = self._read_fortran(outdir / "W", (n_comp, n_comp))
            A = self._read_fortran(outdir / "A", (n_comp, n_comp))
            c = self._read_fortran(outdir / "c", (n_comp,))
            alpha = self._read_fortran(outdir / "alpha", (J, n_comp))
            mu = self._read_fortran(outdir / "mu", (J, n_comp))
            beta = self._read_fortran(outdir / "sbeta", (J, n_comp))
            rho = self._read_fortran(outdir / "rho", (J, n_comp))
            S = self._read_fortran(outdir / "S", (n_comp, n_ch))
            mean_vec = self._read_fortran(outdir / "mean", (n_ch,))

            return {
                "W": W,
                "A": A,
                "alpha": alpha,
                "mu": mu,
                "beta": beta,
                "rho": rho,
                "c": c,
                "ll_history": np.array(ll_history),
                "sphere": S,
                "mean": mean_vec,
                "log_det_sphere": 0.0,
                "elapsed": elapsed,
                "n_iter": len(ll_history),
            }
        finally:
            # Clean up
            if workdir.exists():
                shutil.rmtree(workdir, ignore_errors=True)

    def _write_param_file(self, path, fdt_path, outdir,
                          n_ch, n_samples, n_comp, params, n_iters):
        """Write Fortran AMICA param file using correct field names."""
        lines = [
            f"files {fdt_path}",
            f"outdir {outdir}",
            f"data_dim {n_ch}",
            f"field_dim {n_samples}",
            f"num_models 1",
            f"num_mix_comps {params['num_mix']}",
            f"pcakeep {n_comp}",
            f"max_iter {n_iters}",
            f"max_threads 4",
            f"dble_data 0",
            f"block_size 256",
            f"do_opt_block 0",
            f"lrate {params['lrate']:.6e}",
            f"newtrate {params['newtrate']:.6e}",
            f"newt_start {params['newt_start']}",
            f"newt_ramp {params['newt_ramp']}",
            f"do_newton 1",
            f"rho0 {params['rho0']:.2f}",
            f"minrho {params['minrho']:.2f}",
            f"maxrho {params['maxrho']:.2f}",
            f"rholrate {params['rholrate']:.6e}",
            f"invsigmin {params['invsigmin']:.2e}",
            f"invsigmax {params['invsigmax']:.2e}",
            f"max_decs {params['max_decs']}",
            f"do_reject {'1' if params['do_reject'] else '0'}",
            f"doscaling {'1' if params['doscaling'] else '0'}",
            f"fix_init 1",
            f"writestep 1",
            f"write_nd 1",
            f"write_LLt 0",
            f"do_mean 1",
            f"do_sphere 1",
            f"do_approx_sphere 1",
            f"use_grad_norm {'1' if params.get('use_grad_norm', False) else '0'}",
            f"use_min_dll 1",
            f"min_dll {params['min_dll']:.2e}",
            f"num_samples_start 0",
            f"num_samples_stop {n_samples}",
        ]
        path.write_text("\n".join(lines) + "\n")

    @staticmethod
    def _parse_ll(stdout):
        """Parse LL values from Fortran stdout."""
        lls = []
        for line in stdout.splitlines():
            m = re.search(r"LL\s*=\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)", line)
            if m:
                lls.append(float(m.group(1)))
        return lls

    @staticmethod
    def _read_fortran(path, shape):
        """Read Fortran binary output file (float64, column-major)."""
        if not path.exists():
            return np.zeros(shape)
        raw = np.fromfile(str(path), dtype=np.float64)
        if raw.size == 0:
            return np.zeros(shape)
        return raw.reshape(shape, order="F")
