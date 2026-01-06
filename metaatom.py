"""
KivyMD GUI to explore pretrained neural networks for a cross-shaped metaatom.

The GUI provides:
- Four geometry sliders: Lx1, Ly1, Lx2, Ly2 (normalized to [0.1, 0.9])
- Mode toggles: Asymmetric Cross, Rectangle, Normal Cross
- Live forward prediction of:
  - Reflectivity at 1064 nm and 1550 nm (R1, R2 in [0, 1])
  - Phase at 1064 nm and 1550 nm computed from predicted real/imag parts

UI layout is defined in `metaatom.kv`. 
Model definitions are in `model.py` (PyTorch Lightning).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch

from kivy.core.window import Window
from kivy.lang import Builder
from kivymd.app import MDApp

from model import Tandem, phaseMLP, refMLP 


@dataclass(frozen=True)
class CheckpointPaths:
    """Container for checkpoint file paths."""
    tandem: Path
    forward_real: Path
    forward_imag: Path
    forward_ref: Path


class MainApp(MDApp):
    """
    Main KivyMD application.

    The KV file triggers `app.make_prediction([...])` on slider changes.
    """

    def __init__(
        self,
        checkpoints_dir: str | Path = "checkpoints",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Window defaults (you can change later)
        Window.size = (1000, 400) 

        ckpt_dir = Path(checkpoints_dir)

        self.ckpts = CheckpointPaths(
            tandem=ckpt_dir / "tandemnet.ckpt",
            forward_real=ckpt_dir / "forward_real.ckpt",
            forward_imag=ckpt_dir / "forward_imag.ckpt",
            forward_ref=ckpt_dir / "forward_ref.ckpt",
        )

        # Mode & current parameters (normalized geometric parameters)
        self.mode: str = "Asymmetric Cross"
        self.lx1 = 0.5
        self.ly1 = 0.5
        self.lx2 = 0.5
        self.ly2 = 0.5

        # Models (loaded in _load_models)
        self.tandem: Tandem | None = None
        self.forward_real: phaseMLP | None = None
        self.forward_imag: phaseMLP | None = None
        self.forward_ref: refMLP | None = None

    # ---------------------------------------------------------------------
    # Kivy lifecycle
    # ---------------------------------------------------------------------

    def build(self):
        # Theme
        self.theme_cls.theme_style = "Dark"  
        self.theme_cls.primary_palette = "BlueGray" 

        # Build UI
        root = Builder.load_file("metaatom.kv")  

        # Load models after UI exists (so we can surface errors nicely if desired)
        self._load_models()

        # Trigger one initial draw/prediction from current slider defaults
        try:
            self.make_prediction([self.lx1, self.ly1, self.lx2, self.ly2])
        except Exception:
            # Keep UI alive even if prediction fails (e.g., missing checkpoints)
            pass

        return root

    def _load_models(self) -> None:
        """
        Load pretrained checkpoints and set to eval mode.

        Raises FileNotFoundError with a clear message if checkpoints are missing.
        """
        missing = [p for p in self.ckpts.__dict__.values() if not Path(p).exists()]
        if missing:
            msg = (
                "Missing checkpoint files:\n"
                + "\n".join(f"- {p}" for p in missing)
                + "\n\nExpected layout:\n"
                "checkpoints/\n"
                "  tandemnet.ckpt\n"
                "  forward_real.ckpt\n"
                "  forward_imag.ckpt\n"
                "  forward_ref.ckpt\n"
            )
            raise FileNotFoundError(msg)

        # Load Lightning checkpoints (same as original, just centralized).
        self.tandem = Tandem.load_from_checkpoint(str(self.ckpts.tandem))
        self.forward_real = phaseMLP.load_from_checkpoint(str(self.ckpts.forward_real))
        self.forward_imag = phaseMLP.load_from_checkpoint(str(self.ckpts.forward_imag))
        self.forward_ref = refMLP.load_from_checkpoint(str(self.ckpts.forward_ref))

        # Eval mode once (no need to call eval() every slider move).
        self.tandem.eval()
        self.forward_real.eval()
        self.forward_imag.eval()
        self.forward_ref.eval()

    # ---------------------------------------------------------------------
    # Public methods called from KV
    # ---------------------------------------------------------------------

    def toggleSliders(self) -> None:
        """
        Called by ToggleButtons in the KV file. [file:8]
        Adjusts slider enable/disable pattern and sets self.mode.
        """
        if self.root.ids.rectanglebutton.state == "down":
            # Rectangle: one parameter drives all
            self.root.ids.lx1slider.disabled = False
            self.root.ids.ly1slider.disabled = True
            self.root.ids.lx2slider.disabled = True
            self.root.ids.ly2slider.disabled = True
            self.mode = "Rectangle"

        elif self.root.ids.normalcrossbutton.state == "down":
            # Normal Cross: 2 parameters active
            self.root.ids.lx1slider.disabled = False
            self.root.ids.ly1slider.disabled = False
            self.root.ids.lx2slider.disabled = True
            self.root.ids.ly2slider.disabled = True
            self.mode = "Normal Cross"

        else:
            # Asymmetric Cross: all parameters active
            self.root.ids.lx1slider.disabled = False
            self.root.ids.ly1slider.disabled = False
            self.root.ids.lx2slider.disabled = False
            self.root.ids.ly2slider.disabled = False
            self.mode = "Asymmetric Cross"

        # Update prediction using current slider state
        self.make_prediction(
            [
                self.root.ids.lx1slider.value,
                self.root.ids.ly1slider.value,
                self.root.ids.lx2slider.value,
                self.root.ids.ly2slider.value,
            ]
        )

    def makePrediction(self, value):
        """
        Backwards-compatible wrapper for KV file calls. [file:8]
        """
        self.make_prediction(value)

    def make_prediction(self, value: Iterable[float]) -> None:
        """
        Forward prediction: design -> (R, phase) at two wavelengths.

        KV file passes a 4-list of slider values. [file:8]
        """
        lx1, ly1, lx2, ly2 = [float(v) for v in value]

        # Apply mode constraints (same rules as original).
        if self.mode == "Rectangle":
            lx1 = ly1 = lx2 = ly2 = lx1
        elif self.mode == "Normal Cross":
            lx2 = ly1
            ly2 = lx1

        self.lx1, self.ly1, self.lx2, self.ly2 = lx1, ly1, lx2, ly2

        if self.forward_real is None or self.forward_imag is None or self.forward_ref is None:
            return

        # Ensure (batch, features) shape = (1, 4)
        design = torch.tensor([[lx1, ly1, lx2, ly2]], dtype=torch.float32)

        with torch.no_grad():
            ref = self.forward_ref(design)      # shape (1, 2) 
            imag = self.forward_imag(design)    # shape (1, 2) 
            real = self.forward_real(design)    # shape (1, 2) 

        # Convert to floats
        R1, R2 = float(ref[0, 0].cpu().numpy()), float(ref[0, 1].cpu().numpy())
        real1, real2 = float(real[0, 0].cpu().numpy()), float(real[0, 1].cpu().numpy())
        imag1, imag2 = float(imag[0, 0].cpu().numpy()), float(imag[0, 1].cpu().numpy())

        phase1 = float(np.arctan2(imag1, real1))
        phase2 = float(np.arctan2(imag2, real2))

        self._update_text_and_indicators(phase1, phase2, R1, R2)
        self._update_canvas_geometry()

    def makeDesign(self, value):
        """
        Backwards-compatible wrapper (kept for parity with original).
        """
        self.make_design(value)

    def make_design(self, value: Iterable[float]) -> None:
        """
        Inverse design: desired phases -> predicted geometry via the tandem net,
        then forward-predict R and phase for that geometry. 
        """
        if self.tandem is None or self.forward_real is None or self.forward_imag is None or self.forward_ref is None:
            return

        phi1, phi2 = [float(v) for v in value]
        desired_phases = torch.tensor([[phi1, phi2]], dtype=torch.float32)

        with torch.no_grad():
            params = self.tandem(desired_phases)  # expected shape (1, 4) 

            ref = self.forward_ref(params)
            imag = self.forward_imag(params)
            real = self.forward_real(params)

        # NOTE: original code hard-mapped indices into a symmetric cross.
        # Keep the same mapping here (but with batch-safe indexing). 
        lx1 = float(params[0, 0].cpu().numpy())
        ly1 = float(params[0, 2].cpu().numpy())
        lx2 = float(params[0, 2].cpu().numpy())
        ly2 = float(params[0, 0].cpu().numpy())

        self.lx1, self.ly1, self.lx2, self.ly2 = lx1, ly1, lx2, ly2

        R1, R2 = float(ref[0, 0].cpu().numpy()), float(ref[0, 1].cpu().numpy())
        real1, real2 = float(real[0, 0].cpu().numpy()), float(real[0, 1].cpu().numpy())
        imag1, imag2 = float(imag[0, 0].cpu().numpy()), float(imag[0, 1].cpu().numpy())

        phase1 = float(np.arctan2(imag1, real1))
        phase2 = float(np.arctan2(imag2, real2))

        self._update_text_and_indicators(phase1, phase2, R1, R2)
        self._update_canvas_geometry()

    # ---------------------------------------------------------------------
    # UI updates
    # ---------------------------------------------------------------------

    def _update_text_and_indicators(self, phase1: float, phase2: float, R1: float, R2: float) -> None:
        # phase labels 
        self.root.ids.phase1text.text = f"Phase = {phase1:.2f} rad"
        self.root.ids.phase2text.text = f"Phase = {phase2:.2f} rad"

        # reflectivity bar indicators use size_hint y-component 
        self.root.ids.refind1.size_hint = [0.1, max(0.001, float(R1))]
        self.root.ids.refind2.size_hint = [0.1, max(0.001, float(R2))]

        self.root.ids.reflabel1.text = f"Reflectivity = {R1 * 100:.3f}%"
        self.root.ids.reflabel2.text = f"Reflectivity = {R2 * 100:.3f}%"

    def _update_canvas_geometry(self) -> None:
        """
        Update the two rectangles drawn in the metaatom canvas based on the
        current lx1, ly1, lx2, ly2 values. 
        """
        canvas = self.root.ids.metaatom.canvas

        atom_rect = canvas.get_group("atom")[0]
        box1 = canvas.get_group("box1")[0]
        box2 = canvas.get_group("box2")[0]

        canvas_width, canvas_height = atom_rect.size
        canvas_posx, canvas_posy = atom_rect.pos

        # Rectangle 1
        newlx1 = canvas_width * self.lx1
        newly1 = canvas_height * self.ly1
        box1.size = (newlx1, newly1)
        box1.pos = (
            canvas_posx + canvas_width * 0.5 - newlx1 * 0.5,
            canvas_posy + canvas_height * 0.5 - newly1 * 0.5,
        )

        # Rectangle 2
        newlx2 = canvas_width * self.lx2
        newly2 = canvas_height * self.ly2
        box2.size = (newlx2, newly2)
        box2.pos = (
            canvas_posx + canvas_width * 0.5 - newlx2 * 0.5,
            canvas_posy + canvas_height * 0.5 - newly2 * 0.5,
        )


if __name__ == "__main__":
    # By default, expects checkpoints in ./checkpoints/
    MainApp(checkpoints_dir="checkpoints").run()
