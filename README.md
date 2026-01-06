# Metaatom GUI: Neural Network Response Explorer

This repository contains a small GUI application to interactively explore pretrained neural networks that predict the optical response of a cross-shaped metaatom from geometric parameters.

The app provides sliders for the metaatom geometry and displays predicted reflectivity and phase for two wavelengths (1064 nm and 1550 nm), using pretrained PyTorch Lightning checkpoints.

---

## Repository Contents

| File | Description |
|------|-------------|
| `metaatom.py` | Main KivyMD application (GUI entrypoint) |
| `metaatom.kv` | Kivy layout file defining sliders, canvas drawing, and labels |
| `model.py` | PyTorch Lightning model definitions used to load `.ckpt` checkpoints |
| `checkpoints/` | Please reach out to me to get the checkpiont files |
| `fonts/` | Fonts referenced by the Kivy layout (e.g. `fonts/normal.ttf`) |

---

## Features

- Interactive geometry control via four normalized parameters: `Lx1`, `Ly1`, `Lx2`, `Ly2`
- Mode toggles:
  - Rectangle (one parameter controls all)
  - Normal Cross (two-parameter symmetric coupling)
  - Asymmetric Cross (all parameters independent)
- Live display of:
  - Reflectivity at 1064 nm and 1550 nm
  - Phase at 1064 nm and 1550 nm (computed from predicted real/imag components)
