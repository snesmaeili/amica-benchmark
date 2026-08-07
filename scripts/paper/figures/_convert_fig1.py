from pathlib import Path

import cairosvg

from _paths import out

_HERE = Path(__file__).resolve().parent
_SVG = _HERE / "fig_amica_workflow.svg"

cairosvg.svg2pdf(url=str(_SVG), write_to=str(out("fig_amica_workflow.pdf")))
cairosvg.svg2png(url=str(_SVG), write_to=str(_HERE / "_fig1_preview.png"), output_width=2700)
print("CONVERTED")
