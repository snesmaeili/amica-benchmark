"""Adapter for Fortran AMICA 1.7 binary."""

import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from .base import AmicaAdapter

# amica15ub is statically linked (no MPI needed), works on any node
FORTRAN_BINARY = Path("/home/sesma/refs/sccn-amica/amica15ub")


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

        with tempfile.TemporaryDirectory(prefix="amica_fortran_") as tmpdir:
            tmpdir = Path(tmpdir)
            outdir = tmpdir / "output"
            outdir.mkdir()

            # Write FDT file (Fortran column-major: channels contiguous)
            fdt_path = tmpdir / "input.fdt"
            data.T.astype(np.float32).tofile(fdt_path)

            # Write param file
            param_path = tmpdir / "input.param"
            self._write_param_file(
                param_path, fdt_path, outdir, n_ch, n_samples, params, n_iters
            )

            # Run Fortran binary
            t0 = time.perf_counter()
            env = {**dict(__import__("os").environ)}
            env["OMP_NUM_THREADS"] = "4"
            env["OMP_STACKSIZE"] = "512M"
            try:
                result = subprocess.run(
                    [str(self._binary), str(param_path)],
                    capture_output=True, text=True, timeout=600,
                    env=env,
                )
                elapsed = time.perf_counter() - t0
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                return None

            if result.returncode != 0:
                print(f"Fortran failed: {result.stderr[:500]}")
                return None

            # Parse LL trajectory from stdout
            ll_history = self._parse_ll(result.stdout)

            # Read output files
            n_comp = params.get("pcakeep", n_ch)
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
                "log_det_sphere": 0.0,  # computed from S if needed
                "elapsed": elapsed,
                "n_iter": len(ll_history),
            }

    def _write_param_file(self, path, fdt_path, outdir, n_ch, n_samples,
                          params, n_iters):
        lines = [
            f"files {fdt_path}",
            f"outdir {outdir}",
            f"num_chans {n_ch}",
            f"num_frames {n_samples}",
            f"num_models 1",
            f"num_mix_comps {params['num_mix']}",
            f"max_iter {n_iters}",
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
            f"write_LLt 1",
            f"do_mean 1",
            f"pcakeep {params.get('pcakeep', n_ch)}",
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
        """Read Fortran binary output file."""
        if not path.exists():
            return np.zeros(shape)
        raw = np.fromfile(str(path), dtype=np.float64)
        if raw.size == 0:
            return np.zeros(shape)
        return raw.reshape(shape, order="F")
