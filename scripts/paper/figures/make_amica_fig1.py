#!/usr/bin/env python
"""Figure 1 (AMICA: model -> MNE-compatible estimator) as a clean, Illustrator-editable vector SVG.

Merged opening figure (guideline Fig 1), Nature-Methods grammar, four labelled blocks (A-D):
  A  Generative model       - M-model mixture, x = A^(h)s + c^(h), joint density + one source density (rho family)
  B  Optimization path      - preprocess -> E-step / M-step loop -> natural-grad/quasi-Newton -> convergence
  C  Software architecture  - Raw/Epochs/array -> shared EM core -> JAX-GPU/JAX-CPU/NumPy-CPU (full/chunked) -> MNE ICA + AmicaResult
  D  Validation ladder      - Fortran parity -> MNE/backend parity -> real-EEG MIR -> multi-model (exploratory)
Dominant A (top-left) + C (top-right); B (bottom-left), D (bottom-right); two columns, ~1.67:1.

Notation matches the preprint (x_t, C, T, M, h, z_t, pi^(h), W^(h), s_t^(h), K, alpha/mu/beta/rho,
psi score, p(h|x_t), kappa=T/C^2). Embedded plots are real toy geometry (numpy), not hand-faked.
Run:  python make_amica_fig1.py   ->  fig_amica_workflow.svg
"""
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / "fig_amica_workflow.svg"
W, H = 1360, 812

# ---- muted academic palette; the three AMICA backends share one blue hue (locked style) ----
INK = "#1b1b1b"; SUB = "#555"; HBORD = "#222"
TEAL = "#2b8cbe"; NAVY = "#08519c"; PETROL = "#1f6f78"
PURPLE = "#6a4c93"; OCHRE = "#cf9a2e"; RED = "#c0392b"
B_GPU = "#08519c"; B_CPU = "#3182bd"; B_NPY = "#74add1"    # backends: dark/med/light blue
RAMP = ["#f7fcf0", "#7bccc4", "#0868ac"]                   # EEG heatmap (GnBu)

S = []
def add(s): S.append(s)


# ---------- text + drawing helpers ----------
def _hx(c):
    c = c.lstrip("#"); return tuple(int(c[i:i+2], 16) for i in (0, 2, 4))
def lerp(c1, c2, t):
    a, b = _hx(c1), _hx(c2); return "#%02x%02x%02x" % tuple(int(round(a[i]+(b[i]-a[i])*t)) for i in range(3))
def ramp(v):
    v = float(np.clip(v, 0, 1))
    return lerp(RAMP[0], RAMP[1], v/0.5) if v < 0.5 else lerp(RAMP[1], RAMP[2], (v-0.5)/0.5)
def mapper(x0, y0, w, h, xr, yr):
    (xa, xb), (ya, yb) = xr, yr
    def f(px, py): return (x0 + (px-xa)/(xb-xa)*w, y0 + h - (py-ya)/(yb-ya)*h)
    return f
def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def rt(s, size):
    """_x/_{xx} -> subscript, ^x/^{xx} -> superscript, via tspan baseline shifts (ascii + Greek only)."""
    res = []; i = 0; carry = 0.0; buf = ""
    def flush():
        nonlocal buf, carry
        if buf:
            if abs(carry) > 0.05:
                res.append(f'<tspan dy="{carry:.1f}">{esc(buf)}</tspan>'); carry = 0.0
            else:
                res.append(esc(buf))
        buf = ""
    while i < len(s):
        ch = s[i]
        if ch in "_^" and i+1 < len(s):
            flush()
            if s[i+1] == "{":
                j = s.index("}", i+2); content = s[i+2:j]; i = j+1
            else:
                content = s[i+1]; i += 2
            dy = size*0.30 if ch == "_" else -size*0.42
            res.append(f'<tspan dy="{dy+carry:.1f}" font-size="{size*0.72:.1f}">{esc(content)}</tspan>')
            carry = -dy
        else:
            buf += ch; i += 1
    flush()
    return "".join(res)
def _adv(s, size):
    i = 0; u = 0.0
    while i < len(s):
        ch = s[i]
        if ch in "_^" and i+1 < len(s):
            if s[i+1] == "{":
                j = s.index("}", i+2); content = s[i+2:j]; i = j+1
            else:
                content = s[i+1]; i += 2
            for c in content: u += (0.28 if c == " " else 0.52)*0.72
        else:
            u += 0.28 if ch == " " else 0.52; i += 1
    return u*size
def txt(x, y, s, size=12, fill=INK, anchor="start", weight="normal", style="normal"):
    if ("_" in s or "^" in s) and anchor in ("middle", "end"):
        w = _adv(s, size); x = x - w/2 if anchor == "middle" else x - w
        anchor = "start"
    return (f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" font-style="{style}" font-family="Helvetica, Arial, sans-serif">{rt(s, size)}</text>')
def box(x, y, w, h, rx=4, fill="#fff", stroke=HBORD, sw=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
def arrow(x1, y1, x2, y2, sw=1.4, mk="ah"):
    return f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="{INK}" stroke-width="{sw}" marker-end="url(#{mk})"/>'
def blocklabel(x, y, letter, title):
    add(txt(x, y, letter, size=18, weight="bold"))
    add(txt(x+18, y, title, size=14, weight="bold"))
def trapezoid(x, y, w, h, label, taper=0.0):
    add(f'<polygon points="{x},{y} {x+w},{y+taper} {x+w},{y+h-taper} {x},{y+h}" fill="url(#gray)" stroke="{INK}" stroke-width="1"/>')
    add(txt(x+w/2, y+h/2+6, label, size=16, anchor="middle", fill="#fff", weight="bold"))


# ---------- embedded data plots (real toy geometry) ----------
def draw_heatmap(x0, y0, w, h, nch=10, nt=13, seed=3):
    rng = np.random.default_rng(seed)
    field = rng.random((nch, nt))
    for _ in range(2):
        field = (field + np.roll(field, 1, 0) + np.roll(field, -1, 0) + np.roll(field, 1, 1) + np.roll(field, -1, 1)) / 5.0
    field = (field - field.min()) / (np.ptp(field) + 1e-9)
    cw, ch = w/nt, h/nch
    for i in range(nch):
        for j in range(nt):
            add(f'<rect x="{x0+j*cw:.0f}" y="{y0+i*ch:.0f}" width="{cw+0.6:.0f}" height="{ch+0.6:.0f}" fill="{ramp(field[i,j])}"/>')
    add(box(x0, y0, w, h, rx=0, fill="none", stroke=INK, sw=1.0))

def draw_scatter(x0, y0, w, h, seed=1):
    rng = np.random.default_rng(seed)
    N = 150
    s1 = rng.laplace(size=N); s2 = rng.laplace(size=N)
    A = np.array([[1.0, 0.62], [0.42, 1.0]])
    Xd = A @ np.vstack([s1, s2]); lim = np.percentile(np.abs(Xd), 99)
    add(box(x0, y0, w, h, rx=3, fill="#fff", stroke=INK, sw=1.0))
    pf = mapper(x0+10, y0+10, w-20, h-20, (-lim, lim), (-lim, lim))
    ax = pf(0, 0)
    add(f'<line x1="{x0+10}" y1="{ax[1]:.0f}" x2="{x0+w-10}" y2="{ax[1]:.0f}" stroke="#ccc" stroke-width="0.7"/>')
    add(f'<line x1="{ax[0]:.0f}" y1="{y0+10}" x2="{ax[0]:.0f}" y2="{y0+h-10}" stroke="#ccc" stroke-width="0.7"/>')
    for xv, yv, c in zip(Xd[0], Xd[1], s1):
        col = TEAL if c >= 0 else PURPLE
        sx, sy = pf(np.clip(xv, -lim, lim), np.clip(yv, -lim, lim))
        add(f'<circle cx="{sx:.0f}" cy="{sy:.0f}" r="1.5" fill="{col}" fill-opacity="0.5"/>')
    Ac = A / np.linalg.norm(A, axis=0)
    for j in range(2):
        tip = pf(Ac[0, j]*lim*0.82, Ac[1, j]*lim*0.82)
        add(f'<line x1="{ax[0]:.0f}" y1="{ax[1]:.0f}" x2="{tip[0]:.0f}" y2="{tip[1]:.0f}" stroke="{INK}" stroke-width="2.1" marker-end="url(#ah)"/>')
    add(txt(x0+w-12, ax[1]-5, "x_1", size=10.5, anchor="end", style="italic", fill=SUB))
    add(txt(ax[0]+6, y0+18, "x_2", size=10.5, style="italic", fill=SUB))
    add(txt(x0+w-12, y0+h-9, "a_j = cols of A", size=9.5, anchor="end", fill=INK, style="italic"))

def draw_minidensity(cx, cy, kind, col=PETROL):
    """small density sparkline glyph centred at (cx, cy)."""
    xs = np.linspace(-1, 1, 40)
    if kind == "heavy":
        yv = np.exp(-np.abs(xs/0.28)**0.7)
    elif kind == "lapl":
        yv = np.exp(-np.abs(xs/0.42))
    else:
        yv = np.exp(-(xs/0.5)**2)
    yv = yv/yv.max()
    pts = " ".join(f"{cx-20+ (k/39)*40:.0f},{cy-yv[k]*20:.0f}" for k in range(40))
    add(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.6"/>')
    add(f'<line x1="{cx-20}" y1="{cy}" x2="{cx+20}" y2="{cy}" stroke="#ccc" stroke-width="0.6"/>')

def draw_llcurve(x0, y0, w, h):
    add(box(x0, y0, w, h, rx=3, fill="#fff", stroke="#bbb", sw=0.8))
    t = np.linspace(0, 1, 50); ll = 1-np.exp(-3.6*t)
    f = mapper(x0+6, y0+6, w-12, h-16, (0, 1), (0, 1.02))
    pts = " ".join(f"{px:.0f},{py:.0f}" for px, py in (f(t[k], ll[k]) for k in range(50)))
    add(f'<polyline points="{pts}" fill="none" stroke="#2a9d8f" stroke-width="2"/>')
    add(txt(x0+w/2, y0+h-4, "iteration", size=8.4, anchor="middle", fill=SUB))

def backend_icon(cx, cy, kind, col):
    if kind == "sq_f":
        add(f'<rect x="{cx-6}" y="{cy-6}" width="12" height="12" rx="1.5" fill="{col}"/>')
    elif kind == "sq_o":
        add(f'<rect x="{cx-6}" y="{cy-6}" width="12" height="12" rx="1.5" fill="none" stroke="{col}" stroke-width="1.8"/>')
    else:  # tri_o
        add(f'<polygon points="{cx},{cy-7} {cx+7},{cy+6} {cx-7},{cy+6}" fill="none" stroke="{col}" stroke-width="1.8"/>')


# ================= canvas + defs =================
add(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica, Arial, sans-serif">')
add('<defs>'
    '<linearGradient id="gray" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#dcdcdc"/><stop offset="1" stop-color="#6f6f6f"/></linearGradient>'
    '<marker id="ah" markerWidth="8" markerHeight="8" refX="6.5" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0 0L6.5 3L0 6z" fill="#1b1b1b"/></marker>'
    '<marker id="ahup" markerWidth="9" markerHeight="9" refX="3" refY="2" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 6L3 0L6 6z" fill="#1b1b1b"/></marker>'
    '</defs>')
add(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
# quadrant dividers (hairline)
add(f'<line x1="832" y1="28" x2="832" y2="788" stroke="#e0e0e0" stroke-width="1"/>')
add(f'<line x1="24" y1="410" x2="1336" y2="410" stroke="#e0e0e0" stroke-width="1"/>')


# ================= BLOCK A : GENERATIVE MODEL =================
blocklabel(32, 50, "A", "Generative model")
add(txt(40, 92, "outer mixture — z_t selects 1 of M ICA models", size=11, weight="bold"))
add(f'<circle cx="74" cy="158" r="20" fill="#fff" stroke="{INK}" stroke-width="1.5"/>')
add(txt(74, 162, "z_t", size=12, anchor="middle", style="italic"))
add(txt(74, 204, "z_t ~ Cat(π)", size=10, anchor="middle", fill=SUB, style="italic"))
add(txt(218, 113, "M models", size=9, anchor="middle", fill=SUB, style="italic"))
add(box(168, 122, 132, 52, rx=5, fill="#fff", stroke="#9badc0", sw=0.9))
add(box(160, 130, 132, 52, rx=5, fill="#fff", stroke="#9badc0", sw=0.9))
add(box(152, 138, 132, 52, rx=5, fill="#fff", stroke=INK, sw=1.1))
add(txt(164, 159, "model h", size=11, weight="bold"))
add(txt(164, 177, "W^{(h)}, A^{(h)}, c^{(h)}, π^{(h)}", size=9.5, style="italic"))
add(arrow(94, 161, 150, 163))
add(txt(122, 151, "selects", size=8.5, anchor="middle", fill=SUB, style="italic"))
add(arrow(304, 164, 334, 164))
# --- within model h: sources -> mix A^(h) -> observed x ---
add(txt(342, 92, "within model h — linear mixing", size=11, weight="bold"))
rng = np.random.default_rng(2)
for k, yy in enumerate((142, 164, 186)):
    sig = np.cumsum(rng.normal(0, 1, 34)); sig = (sig-sig.mean())/(np.abs(sig).max()+1e-9)
    pts = " ".join(f"{342+ (j/33)*60:.0f},{yy - sig[j]*8:.0f}" for j in range(34))
    add(f'<polyline points="{pts}" fill="none" stroke="{PETROL}" stroke-width="1.1"/>')
add(txt(342, 206, "sources s_i", size=10, fill=SUB, style="italic"))
add(arrow(408, 164, 432, 164))
trapezoid(436, 138, 52, 52, "A^{(h)}")
add(arrow(492, 164, 518, 164))
add(txt(464, 210, "x_t = A^{(h)} s_t + c^{(h)}", size=11, anchor="middle", style="italic"))
draw_scatter(560, 96, 168, 168)
add(txt(644, 90, "observed  x_t", size=11, anchor="middle", weight="bold"))
# adaptive source density (folded in: one density family + equation)
add(txt(644, 288, "adaptive source density", size=10, anchor="middle", weight="bold"))
for cx, kind, lab in ((588, "gauss", "ρ≈2"), (644, "lapl", "ρ≈1"), (700, "heavy", "ρ<1")):
    draw_minidensity(cx, 322, kind)
    add(txt(cx, 344, lab, size=8.6, anchor="middle", fill=SUB))
add(txt(644, 362, "shape ρ fit per source", size=8.6, anchor="middle", fill=SUB, style="italic"))
# defining equations (left)
add(txt(40, 296, "the model defines a joint density:", size=11, weight="bold"))
add(txt(40, 320, "p(x_t) = Σ_h π^{(h)} |det W^{(h)}| · Π_i p_i(s_i^{(h)})", size=12.5, style="italic"))
add(txt(40, 346, "p_i(s) = Σ_{k=1}^{K} α_{ik} GG(s − μ_{ik}; β_{ik}, ρ_{ik})", size=11.5, style="italic"))
add(txt(40, 372, "M = 1 benchmark path · M > 1 captures non-stationary regimes", size=9.6, fill=SUB))


# ================= BLOCK B : OPTIMIZATION PATH =================
blocklabel(32, 446, "B", "Optimization path  (maximize the data log-likelihood)")
draw_heatmap(40, 500, 74, 140, nch=10, nt=11)
add(txt(40, 494, "EEG  x", size=11, weight="bold"))
add(arrow(116, 568, 150, 568))
add(box(152, 532, 150, 74, rx=6))
add(txt(227, 552, "preprocess", size=11, anchor="middle", weight="bold"))
add(txt(162, 570, "center · ZCA-sphere", size=10, fill=SUB))
add(txt(162, 586, "PCA, keep C ;  κ = T / C^2", size=10, fill=SUB))
add(arrow(304, 568, 338, 568))
# EM loop container
add(box(346, 476, 358, 232, rx=8, fill="#fff", stroke=INK, sw=1.2, dash="6 4"))
add(txt(525, 496, "EM — iterate until ΔLL converges", size=11.5, anchor="middle", weight="bold", fill=INK))
# E-step
add(box(358, 506, 168, 150, rx=6, fill="#fbfbfd", stroke=INK, sw=1.0))
add(txt(442, 526, "E-step", size=12.5, anchor="middle", weight="bold"))
add(txt(368, 548, "s = W^{(h)}(x − c)", size=10.5, style="italic"))
add(txt(368, 568, "responsibilities (K GG)", size=10.5, fill=SUB))
add(txt(368, 588, "posterior p(h|x_t)", size=10.5, fill=SUB))
add(txt(380, 604, "= π^{(h)} p(x_t|h) / Z", size=10, style="italic", fill=SUB))
add(txt(368, 626, "log-Σ-exp likelihood", size=10.5, fill=SUB))
# M-step
add(box(536, 506, 156, 150, rx=6, fill="#fbfbfd", stroke=INK, sw=1.0))
add(txt(614, 526, "M-step", size=12.5, anchor="middle", weight="bold"))
add(txt(546, 548, "density α, μ, β, ρ", size=10.5, style="italic"))
add(txt(546, 568, "W^{(h)}: natural-grad (ψ)", size=10.5, fill=SUB))
add(txt(556, 584, "then quasi-Newton", size=10.5, fill=SUB))
add(txt(556, 600, "  (near optimum)", size=9.5, fill=SUB, style="italic"))
add(txt(546, 622, "weights π · center c", size=10.5, fill=SUB))
add(txt(546, 638, "column rescale", size=10.5, fill=SUB))
add(arrow(526, 556, 536, 556))
add(f'<path d="M614 656 L614 684 L442 684 L442 658" fill="none" stroke="{INK}" stroke-width="1.4" marker-end="url(#ah)"/>')
add(txt(528, 700, "next iteration", size=10, anchor="middle", fill=SUB))
# optional sample rejection
add(box(346, 718, 358, 30, rx=5, fill="#fbfbfb", stroke="#999", sw=1.0, dash="4 3"))
add(txt(525, 737, "optional: likelihood-based sample rejection (off by default)", size=10, anchor="middle", fill="#666"))
# convergence curve
add(txt(716, 500, "log-likelihood", size=10, weight="bold"))
draw_llcurve(716, 510, 96, 84)
add(txt(764, 612, "until ΔLL < ε", size=9.5, anchor="middle", fill=SUB, style="italic"))


# ================= BLOCK C : SOFTWARE ARCHITECTURE =================
blocklabel(854, 50, "C", "Software architecture")
add(txt(862, 86, "one API · three interchangeable backends · identical math", size=11, weight="bold"))
# inputs
for i, lab in enumerate(["mne.io.Raw", "Epochs", "ndarray C×T"]):
    bx = 866 + i*132
    add(box(bx, 104, 120, 30, rx=5, fill="#f4f7fb", stroke=INK, sw=1.0))
    add(txt(bx+60, 123, lab, size=9.8, anchor="middle"))
add(arrow(1058, 134, 1058, 158))
# shared core
add(box(930, 162, 256, 42, rx=6, fill="#eef3f8", stroke=INK, sw=1.3))
add(txt(1058, 180, "AMICA core — EM (model in A/B)", size=10.5, anchor="middle", weight="bold"))
add(txt(1058, 196, "fit math shared across backends", size=9, anchor="middle", fill=SUB, style="italic"))
add(arrow(1058, 204, 1058, 228))
# three backends (same blue hue, differ by icon)
for i, (lab, col, icon, mem) in enumerate([
        ("JAX-GPU", B_GPU, "sq_f", "full / chunked"),
        ("JAX-CPU", B_CPU, "sq_o", "full / chunked"),
        ("NumPy-CPU", B_NPY, "tri_o", "reference")]):
    bx = 874 + i*138
    add(box(bx, 234, 126, 50, rx=6, fill="#fff", stroke=col, sw=1.5))
    backend_icon(bx+16, 259, icon, col)
    add(txt(bx+72, 255, lab, size=10, anchor="middle", weight="bold", fill=col))
    add(txt(bx+72, 272, mem, size=8.4, anchor="middle", fill=SUB))
add(txt(1058, 300, "chunked path bounds peak memory (full = chunked numerically)", size=8.8, anchor="middle", fill=SUB, style="italic"))
add(arrow(1058, 306, 1058, 330))
# outputs
add(box(906, 334, 304, 50, rx=6, fill="#f4f7fb", stroke=INK, sw=1.1))
add(txt(1058, 353, "mne.preprocessing.ICA  +  AmicaResult", size=10.5, anchor="middle", weight="bold"))
add(txt(1058, 371, "W · maps A · LL · posteriors p(h|t)", size=9, anchor="middle", fill=SUB))
add(txt(1058, 400, "feeds ICLabel · dipole fit · standard MNE-Python tooling", size=9.4, anchor="middle", fill=SUB, style="italic"))


# ================= BLOCK D : VALIDATION LADDER =================
blocklabel(854, 446, "D", "Validation ladder")
add(txt(880, 482, "each layer checked before the next claim", size=11, weight="bold"))
rungs = [
    ("Fortran AMICA 1.7 parity", "matched-source |r| > 0.9999"),
    ("MNE + cross-backend parity", "GPU = CPU = NumPy; round-trip"),
    ("Real-EEG MIR advantage", "3 datasets · 96 subjects (Fig. 2)"),
    ("Multi-model regimes", "exploratory, in-sample (Fig. 5)"),
]
y_bot = 742
for i, (title, sub) in enumerate(rungs):
    yy = y_bot - i*52
    xx = 902 + i*12
    add(box(xx, yy-36, 388, 42, rx=6, fill=lerp("#eef3f8", "#cfe0f0", i/3), stroke=INK, sw=1.1))
    add(txt(xx+14, yy-19, f"{i+1}.  {title}", size=10.4, weight="bold"))
    add(txt(xx+14, yy-5, sub, size=8.8, fill=SUB))
# upward scope/confidence arrow on the left
add(f'<line x1="880" y1="744" x2="880" y2="540" stroke="{INK}" stroke-width="1.6" marker-end="url(#ahup)"/>')
add(f'<text x="872" y="616" font-size="9.5" fill="{SUB}" text-anchor="middle" font-style="italic" '
    f'font-family="Helvetica, Arial, sans-serif" transform="rotate(-90 872 616)">increasing scope + confidence</text>')


add('</svg>')
OUT.write_text("\n".join(S) + "\n", encoding="utf-8", newline="\n")
print("wrote", OUT, f"({len(S)} elements, {OUT.stat().st_size} bytes)")
