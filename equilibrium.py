import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Physical constants ────────────────────────────────────────────────────────
kb   = 8.617e-5   # eV / K
m0   = 9.109e-31  # kg
kb_J = 1.381e-23  # J / K
h_J  = 6.626e-34  # J·s

# ─── Material database ─────────────────────────────────────────────────────────
MATERIALS = {
    "Silicon (Si)":            {"Eg": 1.12, "me_r": 1.08,  "mh_r": 0.56},
    "Germanium (Ge)":          {"Eg": 0.66, "me_r": 0.55,  "mh_r": 0.37},
    "Gallium Arsenide (GaAs)": {"Eg": 1.42, "me_r": 0.067, "mh_r": 0.48},
}

# ─── Dopant database ───────────────────────────────────────────────────────────
# Ionization energies (eV) from Sze & Ng "Physics of Semiconductor Devices" 3rd ed.
# Donors  → energy below Ec (ΔEd);   Ed = Ec − ΔEd
# Acceptors → energy above Ev (ΔEa); Ea = Ev + ΔEa
DOPANTS = {
    "Silicon (Si)": {
        "donors": {
            "P  (Phosphorus)": 0.045,
            "As (Arsenic)":    0.054,
            "Sb (Antimony)":   0.039,
        },
        "acceptors": {
            "B  (Boron)":    0.045,
            "Al (Aluminum)": 0.057,
            "Ga (Gallium)":  0.065,
            "In (Indium)":   0.160,
        },
    },
    "Germanium (Ge)": {
        "donors": {
            "P  (Phosphorus)": 0.0120,
            "As (Arsenic)":    0.0127,
            "Sb (Antimony)":   0.0096,
        },
        "acceptors": {
            "B  (Boron)":    0.0104,
            "Al (Aluminum)": 0.0102,
            "Ga (Gallium)":  0.0108,
            "In (Indium)":   0.0112,
        },
    },
    "Gallium Arsenide (GaAs)": {
        "donors": {
            "Si (Silicon, Ga-site)": 0.0058,
            "Se (Selenium)":         0.0059,
            "Te (Tellurium)":        0.0030,
        },
        "acceptors": {
            "C  (Carbon, As-site)": 0.0267,
            "Zn (Zinc)":            0.0307,
            "Mg (Magnesium)":       0.0284,
            "Be (Beryllium)":       0.0280,
        },
    },
}

# ─── Physics helpers ───────────────────────────────────────────────────────────

def effective_dos(T, me_r, mh_r):
    Nc = 2 * (2 * np.pi * me_r * m0 * kb_J * T / h_J**2) ** 1.5 / 1e6
    Nv = 2 * (2 * np.pi * mh_r * m0 * kb_J * T / h_J**2) ** 1.5 / 1e6
    return Nc, Nv

def log_ni_compute(T, Eg, me_r, mh_r):
    Nc, Nv = effective_dos(T, me_r, mh_r)
    return 0.5 * (np.log(Nc) + np.log(Nv)) - Eg / (2.0 * kb * T)

def intrinsic_ni(T, Eg, me_r, mh_r):
    ln = log_ni_compute(T, Eg, me_r, mh_r)
    return float(np.exp(ln)) if ln > -700 else 0.0

def intrinsic_EF_above_Ev(Eg, T, Nc, Nv):
    return Eg / 2 + (kb * T / 2) * np.log(Nv / Nc)

def fermi_dirac(E, EF, T):
    x = np.clip((E - EF) / (kb * T), -500, 500)
    return 1.0 / (1.0 + np.exp(x))

def dos_conduction(E, Ec, me_r):
    g = np.zeros_like(E)
    m = E >= Ec
    g[m] = me_r ** 1.5 * np.sqrt(E[m] - Ec)
    return g

def dos_valence(E, Ev, mh_r):
    g = np.zeros_like(E)
    m = E <= Ev
    g[m] = mh_r ** 1.5 * np.sqrt(Ev - E[m])
    return g

def solve_EF_selfconsistent(Nd, Na, Ed, Ea, Nc, Nv, Ec, Ev, T):
    """Solve n0 + Na⁻ - p0 - Nd⁺ = 0 for EF via bisection (100 iterations).
    Ed/Ea are the dopant levels in eV; pass None if not applicable.
    Returns (EF, n0, p0) — n0/p0 via Boltzmann, so mass action holds exactly."""
    kT = kb * T

    def balance(EF):
        n0  = Nc * np.exp(np.clip((EF - Ec) / kT, -500, 500))
        p0  = Nv * np.exp(np.clip((Ev - EF) / kT, -500, 500))
        Ndp = (Nd / (1.0 + 2.0  * np.exp(np.clip((EF - Ed) / kT, -500, 500)))
               if (Nd > 0 and Ed is not None) else 0.0)
        Nam = (Na / (1.0 + 0.25 * np.exp(np.clip((Ea - EF) / kT, -500, 500)))
               if (Na > 0 and Ea is not None) else 0.0)
        return n0 + Nam - p0 - Ndp

    lo, hi = Ev - 0.5, Ec + 0.5
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if balance(mid) < 0:
            lo = mid
        else:
            hi = mid
    EF = (lo + hi) / 2.0
    kT = kb * T
    n0 = Nc * np.exp(np.clip((EF - Ec) / kT, -500, 500))
    p0 = Nv * np.exp(np.clip((Ev - EF) / kT, -500, 500))
    return EF, n0, p0

def EF_from_log_n(log_n, Nc, Ec, T):
    return Ec + kb * T * (log_n - np.log(Nc))

def EF_from_log_p(log_p, Nv, Ev, T):
    return Ev + kb * T * (np.log(Nv) - log_p)

def ionization_donor(EF, Ed, T):
    """Nd+/Nd = 1 / (1 + 2·exp((EF − Ed) / kT))  — spin degeneracy g=2"""
    x = np.clip((EF - Ed) / (kb * T), -500, 500)
    return 1.0 / (1.0 + 2.0 * np.exp(x))

def ionization_acceptor(EF, Ea, T):
    """Na−/Na = 1 / (1 + 0.25·exp((Ea − EF) / kT))  — valence band degeneracy g=4"""
    x = np.clip((Ea - EF) / (kb * T), -500, 500)
    return 1.0 / (1.0 + 0.25 * np.exp(x))

def fmt_density(v):
    if v <= 0 or np.isnan(v):
        return "~0 cm⁻³"
    return f"{v:.3e} cm⁻³"

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Semiconductor Equilibrium Lab", layout="wide")
st.title("Semiconductor in Equilibrium — Interactive Lab")
st.caption("GUPS2005 Solid State Electronics | Dr. Yi-Chun Chin")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Global Settings")
    material_name = st.selectbox("Material", list(MATERIALS.keys()))
    mat  = MATERIALS[material_name]
    T    = st.slider("Temperature  T (K)", 10, 700, 300, step=10)

    Eg   = mat["Eg"];  me_r = mat["me_r"];  mh_r = mat["mh_r"]
    Ev   = 0.0;        Ec   = Eg

    Nc, Nv   = effective_dos(T, me_r, mh_r)
    log_ni_v = log_ni_compute(T, Eg, me_r, mh_r)
    ni_val   = intrinsic_ni(T, Eg, me_r, mh_r)
    Ei       = Ev + intrinsic_EF_above_Ev(Eg, T, Nc, Nv)

    st.divider()
    st.markdown("**Computed quantities**")
    st.markdown(f"""
| Quantity | Value |
|:--|--:|
| N<sub>c</sub> | {Nc:.2e} cm⁻³ |
| N<sub>v</sub> | {Nv:.2e} cm⁻³ |
| n<sub>i</sub> | {ni_val:.2e} cm⁻³ |
| E<sub>i</sub> | {(Ei-Ev)*1e3:.1f} meV above E<sub>v</sub> |
| k<sub>B</sub>T | {kb*T*1e3:.1f} meV |
""", unsafe_allow_html=True)

E_axis = np.linspace(Ev - 0.6, Ec + 0.6, 3000)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "1 · Fermi-Dirac Distribution",
    "2 · Density of States",
    "3 · Carrier Distribution",
    "4 · Doping & Fermi Level",
])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Fermi-Dirac
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Fermi-Dirac distribution")
    st.latex(r"f_F(E) = \frac{1}{1 + e^{\,(E - E_F)\,/\,k_B T}}")

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        EF_t1      = st.slider("Fermi level EF (eV)",
                               float(Ev - 0.3), float(Ec + 0.3), float(Ei),
                               step=0.005, key="t1_EF")
        show_boltz = st.checkbox("Boltzmann approximation", value=True)
        show_3kT   = st.checkbox("Show \u00b13k\u2082T window", value=True)

        kT = kb * T
        f_Ec   = fermi_dirac(np.array([Ec]), EF_t1, T)[0]
        f_Ev_h = 1 - fermi_dirac(np.array([Ev]), EF_t1, T)[0]
        st.markdown(f"""
**At this E<sub>F</sub>:**
- f(E<sub>c</sub>) = {f_Ec:.3e}
- 1 \u2212 f(E<sub>v</sub>) = {f_Ev_h:.3e}
- k<sub>B</sub>T = {kT*1e3:.1f} meV
""", unsafe_allow_html=True)

    with col_plot:
        f = fermi_dirac(E_axis, EF_t1, T)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=f, y=E_axis, mode="lines", name="Fermi-Dirac f<sub>F</sub>(E)",
            line=dict(color="blue", width=2.5)))

        if show_boltz:
            f_b = np.exp(np.clip(-(E_axis - EF_t1) / (kb * T), -500, 500))
            f_b = np.where(f_b > 2, None, f_b)
            fig1.add_trace(go.Scatter(
                x=f_b, y=E_axis, mode="lines", name="Boltzmann approx.",
                line=dict(color="darkorange", dash="dash", width=1.5)))

        if show_3kT:
            fig1.add_hrect(y0=EF_t1 - 3*kT, y1=EF_t1 + 3*kT,
                           fillcolor="rgba(0,160,0,0.10)", line_width=0,
                           annotation_text="E<sub>F</sub> \u00b1 3k<sub>B</sub>T",
                           annotation_position="top right")

        fig1.add_hrect(y0=Ec,      y1=Ec+0.55, fillcolor="rgba(220,50,50,0.12)",   line_width=0)
        fig1.add_hrect(y0=Ev-0.55, y1=Ev,      fillcolor="rgba(50,130,200,0.12)",  line_width=0)
        fig1.add_hline(y=Ec, line_color="red",   line_width=1.5,
                       annotation_text=f"E<sub>c</sub> = {Ec:.2f} eV",
                       annotation_position="top right", annotation_font_color="red")
        fig1.add_hline(y=Ev, line_color="blue",  line_width=1.5,
                       annotation_text=f"E<sub>v</sub> = {Ev:.2f} eV",
                       annotation_position="bottom right", annotation_font_color="blue")
        fig1.add_hline(y=EF_t1, line_color="green", line_dash="dashdot", line_width=2,
                       annotation_text=f"E<sub>F</sub> = {EF_t1:.3f} eV",
                       annotation_position="right", annotation_font_color="green")

        # Dummy traces so Ec / Ev / EF appear in the legend
        fig1.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
            name=f"E<sub>c</sub> = {Ec:.2f} eV",
            line=dict(color="red", width=1.5)))
        fig1.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
            name=f"E<sub>v</sub> = {Ev:.2f} eV",
            line=dict(color="blue", width=1.5)))
        fig1.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
            name=f"E<sub>F</sub> = {EF_t1:.3f} eV",
            line=dict(color="green", dash="dashdot", width=2)))

        fig1.update_layout(
            xaxis=dict(title="Occupation probability  f(E)", range=[-0.05, 1.25]),
            yaxis=dict(title="Energy (eV)", range=[Ev - 0.5, Ec + 0.5]),
            height=520, margin=dict(r=170, b=120),
            title=f"{material_name}  |  T = {T} K",
            legend=dict(orientation="h", y=-0.18, x=0),
        )
        st.plotly_chart(fig1, use_container_width=False)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Density of States
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Density of states  g(E)")
    st.latex(r"""
        g_c(E) = \frac{4\pi(2m_e^*)^{3/2}}{h^3}\sqrt{E - E_c},\quad
        g_v(E) = \frac{4\pi(2m_h^*)^{3/2}}{h^3}\sqrt{E_v - E}
    """)

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        me_r2 = st.slider("Electron eff. mass  m\u2091*/m\u2080", 0.05, 2.0, float(me_r), step=0.01, key="t2_me")
        mh_r2 = st.slider("Hole eff. mass  m\u1d55*/m\u2080",    0.05, 2.0, float(mh_r), step=0.01, key="t2_mh")
        st.caption(f"Material defaults: m\u2091* = {me_r} m\u2080,  m\u1d55* = {mh_r} m\u2080")
        Nc2, Nv2 = effective_dos(T, me_r2, mh_r2)
        st.markdown(f"""
**Effective DoS at {T} K:**
- N<sub>c</sub> = {Nc2:.2e} cm\u207b\u00b3
- N<sub>v</sub> = {Nv2:.2e} cm\u207b\u00b3
- N<sub>c</sub> / N<sub>v</sub> = {Nc2/Nv2:.3f}
""", unsafe_allow_html=True)

    with col_plot:
        norm2   = max(me_r2, mh_r2) ** 1.5 * np.sqrt(0.5)
        gc2     = dos_conduction(E_axis, Ec, me_r2) / norm2
        gv2     = dos_valence(E_axis,   Ev, mh_r2)  / norm2
        mask_cb = E_axis >= Ec
        mask_vb = E_axis <= Ev

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=gc2[mask_cb], y=E_axis[mask_cb],
            fill="tozerox", fillcolor="rgba(220,50,50,0.45)",
            line=dict(color="red", width=1.5),
            name=f"CB  (m\u2091* = {me_r2:.2f} m\u2080)"))
        fig2.add_trace(go.Scatter(
            x=-gv2[mask_vb], y=E_axis[mask_vb],
            fill="tozerox", fillcolor="rgba(50,130,200,0.45)",
            line=dict(color="blue", width=1.5),
            name=f"VB  (m\u1d55* = {mh_r2:.2f} m\u2080)"))

        fig2.add_vline(x=0, line_color="black", line_width=0.8)
        fig2.add_hrect(y0=Ev, y1=Ec, fillcolor="rgba(255,255,0,0.07)", line_width=0)
        fig2.add_hline(y=Ec, line_color="red",  line_width=1.2,
                       annotation_text="E<sub>c</sub>", annotation_position="top right",
                       annotation_font_color="red")
        fig2.add_hline(y=Ev, line_color="blue", line_width=1.2,
                       annotation_text="E<sub>v</sub>", annotation_position="bottom right",
                       annotation_font_color="blue")

        fig2.update_layout(
            xaxis=dict(title="g(E)  (arb. units)"),
            yaxis=dict(title="Energy (eV)", range=[Ev - 0.5, Ec + 0.5]),
            height=520, margin=dict(r=80, b=80),
            title=f"Density of States \u2014 {material_name}  (E\u1d67 = {Eg} eV)",
            legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig2, use_container_width=False)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Carrier Distribution
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Carrier distribution: n(E) and p(E)")
    st.latex(r"n(E) = g_c(E)\cdot f_F(E), \qquad p(E) = g_v(E)\cdot [1 - f_F(E)]")
    st.markdown("Use the Plotly toolbar (top-right of the plot) to **zoom / pan**; "
                "double-click the plot to **reset** the view.")

    EF_t3 = st.slider("Fermi level EF (eV)",
                      float(Ev - 0.3), float(Ec + 0.3), float(Ei),
                      step=0.005, key="t3_EF")

    norm3    = max(me_r, mh_r) ** 1.5 * np.sqrt(0.5)
    gc3      = dos_conduction(E_axis, Ec, me_r) / norm3
    gv3      = dos_valence(E_axis,   Ev, mh_r)  / norm3
    f3       = fermi_dirac(E_axis, EF_t3, T)
    nE       = gc3 * f3
    pE       = gv3 * (1.0 - f3)
    mask_cb3 = E_axis >= Ec
    mask_vb3 = E_axis <= Ev

    fig3 = make_subplots(
        rows=1, cols=3, shared_yaxes=True,
        subplot_titles=["(A) Density of States",
                        "(B) Fermi-Dirac",
                        "(C) Carrier Distribution = (A)\u00d7(B)"],
        horizontal_spacing=0.04,
    )

    # Panel A — DoS
    fig3.add_trace(go.Scatter(x=gc3[mask_cb3], y=E_axis[mask_cb3],
                              fill="tozerox", fillcolor="rgba(220,50,50,0.40)",
                              line=dict(color="red",  width=1.5),
                              name="CB DoS", showlegend=False), row=1, col=1)
    fig3.add_trace(go.Scatter(x=-gv3[mask_vb3], y=E_axis[mask_vb3],
                              fill="tozerox", fillcolor="rgba(50,130,200,0.40)",
                              line=dict(color="blue", width=1.5),
                              name="VB DoS", showlegend=False), row=1, col=1)
    fig3.add_trace(go.Scatter(x=[0, 0], y=[Ev - 0.5, Ec + 0.5], mode="lines",
                              line=dict(color="black", width=0.8),
                              showlegend=False), row=1, col=1)

    # Panel B — Fermi-Dirac
    fig3.add_trace(go.Scatter(x=f3, y=E_axis, mode="lines",
                              line=dict(color="green", width=2),
                              name="f<sub>F</sub>(E) \u2014 electrons"), row=1, col=2)
    fig3.add_trace(go.Scatter(x=1 - f3, y=E_axis, mode="lines",
                              line=dict(color="purple", dash="dash", width=1.5),
                              name="1\u2212f<sub>F</sub>(E) \u2014 holes"), row=1, col=2)

    # Panel C — n(E), p(E)
    fig3.add_trace(go.Scatter(x=nE[mask_cb3], y=E_axis[mask_cb3],
                              fill="tozerox", fillcolor="rgba(220,50,50,0.70)",
                              line=dict(color="red",  width=1.5),
                              name="n(E) electrons"), row=1, col=3)
    fig3.add_trace(go.Scatter(x=-pE[mask_vb3], y=E_axis[mask_vb3],
                              fill="tozerox", fillcolor="rgba(50,130,200,0.70)",
                              line=dict(color="blue", width=1.5),
                              name="p(E) holes"), row=1, col=3)
    fig3.add_trace(go.Scatter(x=[0, 0], y=[Ev - 0.5, Ec + 0.5], mode="lines",
                              line=dict(color="black", width=0.8),
                              showlegend=False), row=1, col=3)

    # Shared horizontal lines for all three panels
    for col_i in [1, 2, 3]:
        fig3.add_hline(y=Ec,    line_color="red",   line_width=1.2, row=1, col=col_i)
        fig3.add_hline(y=Ev,    line_color="blue",  line_width=1.2, row=1, col=col_i)
        fig3.add_hline(y=EF_t3, line_color="green", line_dash="dashdot", line_width=1.5, row=1, col=col_i)
        fig3.add_hrect(y0=Ev, y1=Ec, fillcolor="rgba(255,255,0,0.07)", line_width=0, row=1, col=col_i)

    fig3.add_annotation(x=0.01, y=Ec + 0.04, xref="x", yref="y",
                        text="<b>E<sub>c</sub></b>", showarrow=False,
                        font=dict(color="red", size=9))
    fig3.add_annotation(x=0.01, y=Ev - 0.07, xref="x", yref="y",
                        text="<b>E<sub>v</sub></b>", showarrow=False,
                        font=dict(color="blue", size=9))
    fig3.add_annotation(x=0.01, y=EF_t3 + 0.04, xref="x", yref="y",
                        text="<b>E<sub>F</sub></b>", showarrow=False,
                        font=dict(color="green", size=9))

    fig3.update_yaxes(title_text="Energy (eV)", range=[Ev - 0.5, Ec + 0.5], row=1, col=1)
    fig3.update_layout(
        height=560,
        title=f"{material_name}  |  T = {T} K  |  E<sub>F</sub> = {EF_t3:.3f} eV",
        legend=dict(orientation="h", y=-0.14),
    )
    st.plotly_chart(fig3, use_container_width=True)

    log_n0    = np.log(Nc) + (EF_t3 - Ec) / (kb * T)
    log_p0    = np.log(Nv) + (Ev  - EF_t3) / (kb * T)
    log_ratio = log_n0 + log_p0 - 2.0 * log_ni_v
    ratio     = float(np.exp(np.clip(log_ratio, -30, 30)))
    n0_t3     = float(np.exp(log_n0)) if log_n0 > -700 else 0.0
    p0_t3     = float(np.exp(log_p0)) if log_p0 > -700 else 0.0

    st.markdown(f"""
| n<sub>0</sub> | p<sub>0</sub> | n<sub>0</sub>·p<sub>0</sub> | n<sub>0</sub>p<sub>0</sub> / n<sub>i</sub>² |
|:--|:--|:--|:--|
| {fmt_density(n0_t3)} | {fmt_density(p0_t3)} | {"~0" if n0_t3 <= 0 or p0_t3 <= 0 else f"{n0_t3*p0_t3:.2e} cm⁻⁶"} | {ratio:.5f} |
""", unsafe_allow_html=True)
    if abs(ratio - 1) < 0.02:
        st.success("n\u2080p\u2080 = n\u1d62\u00b2  \u2713  (mass action law verified)")
    else:
        st.info(f"n\u2080p\u2080 / n\u1d62\u00b2 = {ratio:.4f}  (Boltzmann approximation may be breaking down)")

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Doping & Fermi Level
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Doping & Fermi Level Position")
    st.latex(r"n_0 - p_0 = N_d^+ - N_a^- \quad(\text{charge neutrality}), \qquad n_0 p_0 = n_i^2 \quad(\text{mass action})")

    col_ctrl, col_right = st.columns([1, 2])

    with col_ctrl:
        doping_type = st.radio("Doping", ["Intrinsic", "n-type  (donors)", "p-type  (acceptors)"])

        Nd = Na = 0.0
        dE_dopant    = None   # ionisation energy (eV from band edge)
        E_dopant     = None   # absolute energy level (eV)
        dopant_label = ""     # annotation string

        if "n-type" in doping_type:
            Nd_exp = st.slider("log\u2081\u2080(N\u1d30 / cm\u207b\u00b3)", 10.0, 20.0, 15.0, step=0.05)
            Nd = 10 ** Nd_exp
            donor_opts = list(DOPANTS[material_name]["donors"].keys())
            sel_donor  = st.selectbox("Dopant species", donor_opts, key="t4_donor")
            dE_dopant    = DOPANTS[material_name]["donors"][sel_donor]
            E_dopant     = Ec - dE_dopant
            sym          = sel_donor.split()[0]
            dopant_label = f"E<sub>d</sub> ({sym}):  E<sub>c</sub> \u2212 {dE_dopant*1e3:.1f} meV"
            st.caption(f"N\u1d30 = {Nd:.2e} cm\u207b\u00b3  |  n\u1d62 = {ni_val:.2e} cm\u207b\u00b3")

        elif "p-type" in doping_type:
            Na_exp = st.slider("log\u2081\u2080(N\u1d2c / cm\u207b\u00b3)", 10.0, 20.0, 15.0, step=0.05)
            Na = 10 ** Na_exp
            acc_opts  = list(DOPANTS[material_name]["acceptors"].keys())
            sel_acc   = st.selectbox("Dopant species", acc_opts, key="t4_acc")
            dE_dopant    = DOPANTS[material_name]["acceptors"][sel_acc]
            E_dopant     = Ev + dE_dopant
            sym          = sel_acc.split()[0]
            dopant_label = f"E<sub>a</sub> ({sym}):  E<sub>v</sub> + {dE_dopant*1e3:.1f} meV"
            st.caption(f"N\u1d2c = {Na:.2e} cm\u207b\u00b3  |  n\u1d62 = {ni_val:.2e} cm\u207b\u00b3")

        # Solve for EF self-consistently (accounts for partial ionisation / freeze-out)
        if Nd == 0 and Na == 0:
            EF_d = Ei
            log_n0_d = log_p0_d = log_ni_v
        else:
            Ed_sc = E_dopant if "n-type" in doping_type else None
            Ea_sc = E_dopant if "p-type" in doping_type else None
            EF_d, _, _ = solve_EF_selfconsistent(Nd, Na, Ed_sc, Ea_sc, Nc, Nv, Ec, Ev, T)
            EF_d = np.clip(EF_d, Ev - 0.3, Ec + 0.3)
            log_n0_d = np.log(Nc) + (EF_d - Ec) / (kb * T)
            log_p0_d = np.log(Nv) + (Ev - EF_d) / (kb * T)

        n0_d_val  = float(np.exp(log_n0_d)) if log_n0_d > -700 else 0.0
        p0_d_val  = float(np.exp(log_p0_d)) if log_p0_d > -700 else 0.0
        ratio_d   = float(np.exp(np.clip(log_n0_d + log_p0_d - 2.0*log_ni_v, -30, 30)))

        # Ionization fraction (only when a specific dopant species is selected)
        ion_row = ""
        if dE_dopant is not None and E_dopant is not None:
            if "n-type" in doping_type:
                ion_frac = ionization_donor(EF_d, E_dopant, T)
                ion_row  = f"| N<sub>d</sub>\u207a / N<sub>d</sub> | {ion_frac*100:.2f}&thinsp;% |"
            else:
                ion_frac = ionization_acceptor(EF_d, E_dopant, T)
                ion_row  = f"| N<sub>a</sub>\u207b / N<sub>a</sub> | {ion_frac*100:.2f}&thinsp;% |"

        st.divider()
        st.markdown("**Results**")
        st.markdown(f"""
| Quantity | Value |
|:--|--:|
| n<sub>0</sub> | {fmt_density(n0_d_val)} |
| p<sub>0</sub> | {fmt_density(p0_d_val)} |
| E<sub>F</sub> | {EF_d:.4f} eV |
| E<sub>F</sub> \u2212 E<sub>i</sub> | {(EF_d - Ei)*1e3:.1f} meV |
| n<sub>0</sub>p<sub>0</sub> | {n0_d_val*p0_d_val:.2e} cm\u207b\u2076 |
| n<sub>i</sub>\u00b2 | {ni_val**2:.2e} cm\u207b\u2076 |
{ion_row}
""", unsafe_allow_html=True)

        if abs(ratio_d - 1) < 0.02:
            st.success("n\u2080p\u2080 \u2248 n\u1d62\u00b2  \u2713")
        else:
            st.warning(f"n\u2080p\u2080 / n\u1d62\u00b2 = {ratio_d:.4f}")

    # ── Right column: plots ────────────────────────────────────────────────────
    with col_right:

        # ── Band diagram ───────────────────────────────────────────────────────
        fig_bd = go.Figure()
        fig_bd.add_hrect(y0=Ec,       y1=Ec + 0.4, fillcolor="rgba(220,50,50,0.20)",  line_width=0)
        fig_bd.add_hrect(y0=Ev - 0.4, y1=Ev,        fillcolor="rgba(50,130,200,0.20)", line_width=0)

        fig_bd.add_hline(y=Ec, line_color="red",   line_width=2,
                         annotation_text="E<sub>c</sub>",
                         annotation_position="top right", annotation_font_color="red", annotation_font_size=12)
        fig_bd.add_hline(y=Ev, line_color="blue",  line_width=2,
                         annotation_text="E<sub>v</sub>",
                         annotation_position="bottom right", annotation_font_color="blue", annotation_font_size=12)
        fig_bd.add_hline(y=Ei, line_color="gray",  line_width=1.5, line_dash="dash",
                         annotation_text="E<sub>i</sub>",
                         annotation_position="right", annotation_font_color="gray", annotation_font_size=11)
        fig_bd.add_hline(y=EF_d, line_color="green", line_width=2.5, line_dash="dashdot",
                         annotation_text=f"E<sub>F</sub> = {EF_d:.3f} eV",
                         annotation_position="right", annotation_font_color="green", annotation_font_size=12)

        if E_dopant is not None:
            d_color = "purple" if "n-type" in doping_type else "saddlebrown"
            fig_bd.add_hline(y=E_dopant, line_color=d_color, line_width=1.8, line_dash="solid",
                             annotation_text=dopant_label,
                             annotation_position="right",
                             annotation_font_color=d_color, annotation_font_size=11)

        if abs(EF_d - Ei) > 0.003:
            mid = (EF_d + Ei) / 2
            fig_bd.add_shape(type="line", xref="paper", yref="y",
                             x0=0.12, x1=0.12,
                             y0=min(EF_d, Ei), y1=max(EF_d, Ei),
                             line=dict(color="purple", width=1.5))
            fig_bd.add_annotation(xref="paper", yref="y", x=0.15, y=mid,
                                  text=f"\u0394E = {(EF_d-Ei)*1e3:+.1f} meV",
                                  showarrow=False, xanchor="left",
                                  font=dict(color="purple", size=10))

        fig_bd.update_layout(
            xaxis=dict(visible=False, range=[0, 1]),
            yaxis=dict(title="Energy (eV)", range=[Ev - 0.45, Ec + 0.45]),
            height=440, margin=dict(r=220, l=60),
            title=f"{material_name}  |  T = {T} K",
            showlegend=False,
        )
        st.plotly_chart(fig_bd, use_container_width=True)

        # ── Sweep: EF and carrier densities vs doping (self-consistent) ──────────
        N_sweep  = np.logspace(10, 20, 400)
        EF_sweep, n_sweep, p_sweep = [], [], []
        for Ndop in N_sweep:
            if "n-type" in doping_type:
                ef_s, ns, ps = solve_EF_selfconsistent(Ndop, 0.0, E_dopant, None, Nc, Nv, Ec, Ev, T)
            elif "p-type" in doping_type:
                ef_s, ns, ps = solve_EF_selfconsistent(0.0, Ndop, None, E_dopant, Nc, Nv, Ec, Ev, T)
            else:
                ef_s = Ei; ns = ps = ni_val
            EF_sweep.append(float(np.clip(ef_s, Ev - 0.35, Ec + 0.35)))
            n_sweep.append(max(ns, 1e-40))
            p_sweep.append(max(ps, 1e-40))

        fig_sw = go.Figure()
        fig_sw.add_hrect(y0=Ev, y1=Ec, fillcolor="rgba(255,255,0,0.06)", line_width=0)
        fig_sw.add_trace(go.Scatter(x=N_sweep, y=EF_sweep, mode="lines",
                                    name="E<sub>F</sub>", line=dict(color="green", width=2)))
        fig_sw.add_hline(y=Ec, line_color="red",  line_width=1.2,
                         annotation_text="E<sub>c</sub>", annotation_position="top right",
                         annotation_font_color="red")
        fig_sw.add_hline(y=Ev, line_color="blue", line_width=1.2,
                         annotation_text="E<sub>v</sub>", annotation_position="bottom right",
                         annotation_font_color="blue")
        fig_sw.add_hline(y=Ei, line_color="gray", line_dash="dot",  line_width=1.0,
                         annotation_text="E<sub>i</sub>", annotation_position="right",
                         annotation_font_color="gray")
        if E_dopant is not None:
            d_color = "purple" if "n-type" in doping_type else "saddlebrown"
            fig_sw.add_hline(y=E_dopant, line_color=d_color, line_dash="solid", line_width=1.2,
                             annotation_text=dopant_label,
                             annotation_position="right", annotation_font_color=d_color)

        if "n-type" in doping_type:
            fig_sw.add_vline(x=Nd, line_color="orange", line_dash="dashdot", line_width=1.5)
            fig_sw.add_trace(go.Scatter(x=[Nd], y=[EF_d], mode="markers",
                                        marker=dict(color="orange", size=9),
                                        name=f"current N\u1d30 = {Nd:.1e}"))
        elif "p-type" in doping_type:
            fig_sw.add_vline(x=Na, line_color="orange", line_dash="dashdot", line_width=1.5)
            fig_sw.add_trace(go.Scatter(x=[Na], y=[EF_d], mode="markers",
                                        marker=dict(color="orange", size=9),
                                        name=f"current N\u1d2c = {Na:.1e}"))

        lbl = "N\u1d30" if "n-type" in doping_type else ("N\u1d2c" if "p-type" in doping_type else "N")
        fig_sw.update_xaxes(type="log", title="Doping concentration (cm\u207b\u00b3)", exponentformat="power")
        fig_sw.update_yaxes(title="E<sub>F</sub> (eV)", range=[Ev - 0.35, Ec + 0.35])
        fig_sw.update_layout(height=340, margin=dict(r=210, b=70),
                             title=f"E<sub>F</sub> vs. {lbl}",
                             legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5))
        st.plotly_chart(fig_sw, use_container_width=True)

        # ── Carrier densities vs doping ────────────────────────────────────────
        plot_floor = max(ni_val * 1e-3, 1e-40)
        n_arr = np.maximum(np.array(n_sweep), plot_floor)
        p_arr = np.maximum(np.array(p_sweep), plot_floor)

        fig_cd = go.Figure()
        fig_cd.add_trace(go.Scatter(x=N_sweep, y=n_arr, mode="lines",
                                    name="n\u2080 (electrons)", line=dict(color="red", width=2)))
        fig_cd.add_trace(go.Scatter(x=N_sweep, y=p_arr, mode="lines",
                                    name="p\u2080 (holes)", line=dict(color="blue", width=2)))
        if ni_val > 0:
            fig_cd.add_hline(y=ni_val, line_color="gray", line_dash="dot", line_width=1.2,
                             annotation_text=f"n\u1d62 = {ni_val:.1e}",
                             annotation_position="right", annotation_font_color="gray")
        if "n-type" in doping_type and Nd > 0:
            fig_cd.add_vline(x=Nd, line_color="orange", line_dash="dashdot", line_width=1.5)
        elif "p-type" in doping_type and Na > 0:
            fig_cd.add_vline(x=Na, line_color="orange", line_dash="dashdot", line_width=1.5)

        fig_cd.update_xaxes(type="log", title="Doping concentration (cm\u207b\u00b3)", exponentformat="power")
        fig_cd.update_yaxes(type="log", title="Carrier density (cm\u207b\u00b3)", exponentformat="power")
        fig_cd.update_layout(height=320, margin=dict(r=120, b=70),
                             title="n\u2080 and p\u2080  (n\u2080\u00b7p\u2080 = n\u1d62\u00b2 always)",
                             legend=dict(orientation="h", yanchor="top", y=-0.30, xanchor="center", x=0.5))
        st.plotly_chart(fig_cd, use_container_width=True)
