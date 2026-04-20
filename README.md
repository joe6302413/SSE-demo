# SSE Demo — Semiconductor Equilibrium Interactive Lab

An interactive educational tool for **GUPS2005 Solid State Electronics** at National Taiwan University. Students can explore the physics of semiconductors in equilibrium through live visualizations — no equations to stare at, just sliders and plots.

## What it covers

| Tab | Physics |
|:----|:--------|
| **1 · Fermi-Dirac Distribution** | f(E) curve, Boltzmann approximation, ±3k_BT window |
| **2 · Density of States** | Parabolic DoS in conduction and valence bands, effective mass dependence |
| **3 · Carrier Distribution** | n(E) = g_c(E)·f(E) and p(E) = g_v(E)·[1−f(E)], mass action law n₀p₀ = nᵢ² |
| **4 · Doping & Fermi Level** | Charge neutrality solver, E_F vs. doping concentration sweep, n₀/p₀ vs. doping |

All tabs respond to a global **Material** (Si, Ge, GaAs) and **Temperature** (100–700 K) selector in the sidebar.

## How to run

The only prerequisite is [uv](https://docs.astral.sh/uv/getting-started/installation/) — a fast Python package manager that handles everything else automatically.

**Step 1 — Install uv** (one-time, copy the command for your OS from [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/))

**Step 2 — Clone this repository**

```bash
git clone https://github.com/joe6302413/SSE-demo.git
cd SSE-demo
```

**Step 3 — Run the app**

```bash
uv run streamlit run equilibrium.py
```

`uv` will automatically create a virtual environment and install all dependencies (`streamlit`, `numpy`, `matplotlib`) on the first run. After that, the app opens in your browser at `http://localhost:8501`.

## Dependencies

Managed automatically by `uv` via `pyproject.toml`:

- [Streamlit](https://streamlit.io/) — interactive web UI
- [NumPy](https://numpy.org/) — numerical computation
- [Matplotlib](https://matplotlib.org/) — plots

## Course

**GUPS2005 Solid State Electronics**
Dr. Yi-Chun Chin — Global Undergraduate Programs in Semiconductors, National Taiwan University

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).  
You are free to use, modify, and distribute this software under the same license.
