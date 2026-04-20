import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

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

# ─── Physics helpers ───────────────────────────────────────────────────────────

def effective_dos(T, me_r, mh_r):
    """Nc, Nv in cm⁻³ via the parabolic-band formula."""
    Nc = 2 * (2 * np.pi * me_r * m0 * kb_J * T / h_J**2) ** 1.5 / 1e6
    Nv = 2 * (2 * np.pi * mh_r * m0 * kb_J * T / h_J**2) ** 1.5 / 1e6
    return Nc, Nv

def log_ni_compute(T, Eg, me_r, mh_r):
    """ln(ni) computed in log-space — safe at any temperature (no underflow)."""
    Nc, Nv = effective_dos(T, me_r, mh_r)
    return 0.5 * (np.log(Nc) + np.log(Nv)) - Eg / (2.0 * kb * T)

def intrinsic_ni(T, Eg, me_r, mh_r):
    ln = log_ni_compute(T, Eg, me_r, mh_r)
    return float(np.exp(ln)) if ln > -700 else 0.0

def intrinsic_EF_above_Ev(Eg, T, Nc, Nv):
    """Ei above Ev: Eg/2 + (kT/2)*ln(Nv/Nc)."""
    return Eg / 2 + (kb * T / 2) * np.log(Nv / Nc)

def fermi_dirac(E, EF, T):
    x = np.clip((E - EF) / (kb * T), -500, 500)
    return 1.0 / (1.0 + np.exp(x))

def dos_conduction(E, Ec, me_r):
    g = np.zeros_like(E)
    m = E >= Ec
    g[m] = me_r**1.5 * np.sqrt(E[m] - Ec)
    return g

def dos_valence(E, Ev, mh_r):
    g = np.zeros_like(E)
    m = E <= Ev
    g[m] = mh_r**1.5 * np.sqrt(Ev - E[m])
    return g

def solve_doping(Nd, Na, ni_val):
    """Charge neutrality + mass action, no clamping."""
    D = Nd - Na
    n = D / 2.0 + np.sqrt((D / 2.0)**2 + ni_val**2)
    p = ni_val**2 / n if n > 0 else 0.0
    return n, p

def EF_from_log_n(log_n, Nc, Ec, T):
    return Ec + kb * T * (log_n - np.log(Nc))

def EF_from_log_p(log_p, Nv, Ev, T):
    return Ev + kb * T * (np.log(Nv) - log_p)

def EF_stable(n, p, Nc, Nv, Ec, Ev, T, log_ni_v):
    """EF via log-space: avoids log(0) when n or p are extremely small."""
    if n > 0 and p > 0:
        log_n = np.log(n)
        log_p = np.log(p)
    else:
        # Fall back to intrinsic midgap
        log_n = log_ni_v
        log_p = log_ni_v
    if n >= p:
        return EF_from_log_n(log_n, Nc, Ec, T)
    else:
        return EF_from_log_p(log_p, Nv, Ev, T)

def fmt_density(v):
    """Format carrier density for display, handles very small values."""
    if v <= 0 or np.isnan(v):
        return "~0 cm⁻³"
    return f"{v:.3e} cm⁻³"

def band_diagram_style(ax, Ev, Ec, EF, Ei, title=""):
    x = [0, 1]
    ax.fill_between(x, Ec, Ec + 0.4, color="tomato",    alpha=0.25, label="Conduction band")
    ax.fill_between(x, Ev - 0.4, Ev, color="steelblue", alpha=0.25, label="Valence band")
    ax.fill_between(x, Ev, Ec,        color="lightyellow", alpha=0.5,  label=f"Gap ({Ec-Ev:.2f} eV)")
    ax.axhline(Ec,  color="red",   lw=2,   ls="-")
    ax.axhline(Ev,  color="blue",  lw=2,   ls="-")
    ax.axhline(Ei,  color="gray",  lw=1.5, ls="--")
    ax.axhline(EF,  color="green", lw=2.5, ls="-.")
    ax.text(1.03, Ec,  "$E_c$",  color="red",   va="center", fontsize=10)
    ax.text(1.03, Ev,  "$E_v$",  color="blue",  va="center", fontsize=10)
    ax.text(1.03, Ei,  "$E_i$",  color="gray",  va="center", fontsize=9)
    ax.text(1.03, EF,  "$E_F$",  color="green", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1.35)
    ax.set_ylim(Ev - 0.45, Ec + 0.45)
    ax.set_xticks([])
    ax.set_ylabel("Energy (eV)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    if title:
        ax.set_title(title, fontsize=10)

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Semiconductor Equilibrium Lab", layout="wide")
st.title("Semiconductor in Equilibrium — Interactive Lab")
st.caption("GUPS2005 Solid State Electronics | Dr. Yi-Chun Chin")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Global Settings")
    material_name = st.selectbox("Material", list(MATERIALS.keys()))
    mat  = MATERIALS[material_name]
    T    = st.slider("Temperature  T (K)", 100, 700, 300, step=10)

    Eg   = mat["Eg"]
    me_r = mat["me_r"]
    mh_r = mat["mh_r"]
    Ev   = 0.0
    Ec   = Eg

    Nc, Nv  = effective_dos(T, me_r, mh_r)
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

# shared energy axis
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
    st.markdown(
        "Move **E<sub>F</sub>** and **T** to see how the occupation probability changes. "
        "The Boltzmann approximation is valid when |E − E<sub>F</sub>| ≫ k<sub>B</sub>T.",
        unsafe_allow_html=True,
    )

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        EF_t1 = st.slider(
            "Fermi level EF (eV)",
            float(Ev - 0.3), float(Ec + 0.3), float(Ei), step=0.005,
            key="t1_EF",
        )
        show_boltz = st.checkbox("Boltzmann approximation", value=True)
        show_3kT   = st.checkbox("Show ±3k\u2082T window", value=True)

        kT = kb * T
        f_at_Ec      = fermi_dirac(np.array([Ec]), EF_t1, T)[0]
        f_at_Ev_hole = 1 - fermi_dirac(np.array([Ev]), EF_t1, T)[0]
        st.markdown(f"""
**At this E<sub>F</sub>:**
- f(E<sub>c</sub>) = {f_at_Ec:.3e}
- 1 − f(E<sub>v</sub>) = {f_at_Ev_hole:.3e}
- k<sub>B</sub>T = {kT*1e3:.1f} meV
""", unsafe_allow_html=True)

    with col_plot:
        fig, ax = plt.subplots(figsize=(5, 6))
        f = fermi_dirac(E_axis, EF_t1, T)
        ax.plot(f, E_axis, "b-", lw=2.5, label=r"Fermi-Dirac $f_F(E)$")

        if show_boltz:
            f_b = np.exp(np.clip(-(E_axis - EF_t1) / (kb * T), -500, 500))
            f_b = np.where(f_b > 2, np.nan, f_b)
            ax.plot(f_b, E_axis, color="darkorange", ls="--", lw=1.5, alpha=0.9, label="Boltzmann approx.")

        if show_3kT:
            ax.axhspan(EF_t1 - 3*kb*T, EF_t1 + 3*kb*T,
                       alpha=0.12, color="green", label=r"$E_F \pm 3k_BT$")

        ax.axhline(Ec,    color="red",   lw=1.5, ls="--", label=f"$E_c$ = {Ec:.2f} eV")
        ax.axhline(Ev,    color="blue",  lw=1.5, ls="--", label=f"$E_v$ = {Ev:.2f} eV")
        ax.axhline(EF_t1, color="green", lw=2,   ls="-.", label=f"$E_F$ = {EF_t1:.3f} eV")
        ax.axhspan(Ev, Ec, alpha=0.06, color="yellow")

        ax.set_xlabel("Occupation probability  f(E)", fontsize=11)
        ax.set_ylabel("Energy (eV)", fontsize=11)
        ax.set_xlim(-0.05, 1.25)
        ax.set_ylim(Ev - 0.5, Ec + 0.5)
        ax.legend(fontsize=8, loc="center right")
        ax.grid(alpha=0.3)
        ax.set_title(f"{material_name}  |  T = {T} K", fontsize=10)
        st.pyplot(fig, use_container_width=False)
        plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Density of States
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Density of states  g(E)")
    st.latex(r"""
        g_c(E) = \frac{4\pi(2m_e^*)^{3/2}}{h^3}\sqrt{E - E_c},\quad
        g_v(E) = \frac{4\pi(2m_h^*)^{3/2}}{h^3}\sqrt{E_v - E}
    """)
    st.markdown(
        "Adjust the effective masses to see how the parabolic DoS shape changes. "
        "A heavier effective mass → more available states → wider parabola."
    )

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        me_r2 = st.slider("Electron eff. mass  mₑ*/m₀", 0.05, 2.0, float(me_r), step=0.01, key="t2_me")
        mh_r2 = st.slider("Hole eff. mass  mₕ*/m₀",    0.05, 2.0, float(mh_r), step=0.01, key="t2_mh")
        st.caption(f"Material defaults: mₑ* = {me_r} m₀,  mₕ* = {mh_r} m₀")

        Nc2, Nv2 = effective_dos(T, me_r2, mh_r2)
        st.markdown(f"""
**Effective DoS at {T} K:**
- N<sub>c</sub> = {Nc2:.2e} cm⁻³
- N<sub>v</sub> = {Nv2:.2e} cm⁻³
- N<sub>c</sub> / N<sub>v</sub> = {Nc2/Nv2:.3f}
""", unsafe_allow_html=True)

    with col_plot:
        norm2 = max(me_r2, mh_r2)**1.5 * np.sqrt(0.5)
        gc2   = dos_conduction(E_axis, Ec, me_r2) / norm2
        gv2   = dos_valence(E_axis, Ev, mh_r2)    / norm2

        fig, ax = plt.subplots(figsize=(5, 6))
        ax.fill_betweenx(E_axis, 0,   gc2, where=(E_axis >= Ec),
                         color="tomato",    alpha=0.5, label=f"CB  ($m_e^*$={me_r2:.2f} $m_0$)")
        ax.fill_betweenx(E_axis, 0,  -gv2, where=(E_axis <= Ev),
                         color="steelblue", alpha=0.5, label=f"VB  ($m_h^*$={mh_r2:.2f} $m_0$)")
        ax.plot( gc2, E_axis, "r-",  lw=1.5)
        ax.plot(-gv2, E_axis, "b-",  lw=1.5)
        ax.axhline(Ec, color="red",  lw=1.2, ls="--", label="$E_c$")
        ax.axhline(Ev, color="blue", lw=1.2, ls="--", label="$E_v$")
        ax.axhspan(Ev, Ec, alpha=0.07, color="yellow")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("g(E)  (arb. units)", fontsize=11)
        ax.set_ylabel("Energy (eV)", fontsize=11)
        ax.set_ylim(Ev - 0.5, Ec + 0.5)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_title(f"Density of States — {material_name}  ($E_g$ = {Eg} eV)", fontsize=10)
        st.pyplot(fig, use_container_width=False)
        plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Carrier Distribution
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Carrier distribution: n(E) and p(E)")
    st.latex(r"""
        n(E) = g_c(E)\cdot f_F(E), \qquad
        p(E) = g_v(E)\cdot [1 - f_F(E)]
    """)
    st.markdown(
        "The **shaded area** under each curve equals the total carrier density. "
        "The Boltzmann approximation is valid when E<sub>F</sub> is at least 3k<sub>B</sub>T inside the gap.",
        unsafe_allow_html=True,
    )

    EF_t3 = st.slider(
        "Fermi level EF (eV)",
        float(Ev - 0.3), float(Ec + 0.3), float(Ei), step=0.005, key="t3_EF",
    )

    norm3 = max(me_r, mh_r)**1.5 * np.sqrt(0.5)
    gc3   = dos_conduction(E_axis, Ec, me_r) / norm3
    gv3   = dos_valence(E_axis, Ev, mh_r)    / norm3
    f3    = fermi_dirac(E_axis, EF_t3, T)
    nE    = gc3 * f3
    pE    = gv3 * (1.0 - f3)

    fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharey=True)

    # Panel A — DoS
    axes[0].fill_betweenx(E_axis, 0,   gc3, where=(E_axis >= Ec), color="tomato",    alpha=0.4)
    axes[0].fill_betweenx(E_axis, 0,  -gv3, where=(E_axis <= Ev), color="steelblue", alpha=0.4)
    axes[0].plot( gc3, E_axis, "r-",  lw=1.5)
    axes[0].plot(-gv3, E_axis, "b-",  lw=1.5)
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel("g(E)  (arb.)", fontsize=10)
    axes[0].set_ylabel("Energy (eV)", fontsize=11)
    axes[0].set_title("(A) Density of States", fontsize=10)

    # Panel B — Fermi-Dirac
    axes[1].plot(f3,       E_axis, "g-",  lw=2,   label=r"$f_F(E)$ — electrons")
    axes[1].plot(1.0 - f3, E_axis, "m--", lw=1.5, label=r"$1-f_F(E)$ — holes")
    axes[1].set_xlabel("Occupation probability", fontsize=10)
    axes[1].set_title("(B) Fermi-Dirac", fontsize=10)
    axes[1].set_xlim(-0.05, 1.15)
    axes[1].legend(fontsize=8)

    # Panel C — n(E) and p(E)
    axes[2].fill_betweenx(E_axis, 0,   nE, where=(E_axis >= Ec), color="tomato",    alpha=0.7, label="n(E) electrons")
    axes[2].fill_betweenx(E_axis, 0,  -pE, where=(E_axis <= Ev), color="steelblue", alpha=0.7, label="p(E) holes")
    axes[2].plot( nE, E_axis, "r-",  lw=1.5)
    axes[2].plot(-pE, E_axis, "b-",  lw=1.5)
    axes[2].axvline(0, color="k", lw=0.8)
    axes[2].set_xlabel("n(E), p(E)  (arb.)", fontsize=10)
    axes[2].set_title("(C) Carrier Distribution\n= (A) \u00d7 (B)", fontsize=10)
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.axhline(Ec,    color="red",   lw=1.2, ls="--")
        ax.axhline(Ev,    color="blue",  lw=1.2, ls="--")
        ax.axhline(EF_t3, color="green", lw=1.5, ls="-.", alpha=0.9)
        ax.axhspan(Ev, Ec, alpha=0.07, color="yellow")
        ax.set_ylim(Ev - 0.5, Ec + 0.5)
        ax.grid(alpha=0.3)

    axes[0].text(0.01, Ec + 0.03, "$E_c$", color="red",   fontsize=9)
    axes[0].text(0.01, Ev - 0.08, "$E_v$", color="blue",  fontsize=9)
    axes[0].text(0.01, EF_t3 + 0.03, "$E_F$", color="green", fontsize=9)

    fig.suptitle(f"{material_name}  |  T = {T} K  |  $E_F$ = {EF_t3:.3f} eV", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Carrier density metrics — computed in log-space to avoid clamp errors ──
    log_n0 = np.log(Nc) + (EF_t3 - Ec) / (kb * T)
    log_p0 = np.log(Nv) + (Ev  - EF_t3) / (kb * T)
    # log(n0·p0 / ni²) = log_n0 + log_p0 − 2·log_ni  (= 0 by mass-action identity)
    log_ratio = log_n0 + log_p0 - 2.0 * log_ni_v
    ratio = float(np.exp(np.clip(log_ratio, -30, 30)))

    n0_t3 = float(np.exp(log_n0)) if log_n0 > -700 else 0.0
    p0_t3 = float(np.exp(log_p0)) if log_p0 > -700 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("n₀ (electrons)", fmt_density(n0_t3))
    c2.metric("p₀ (holes)",     fmt_density(p0_t3))
    c3.metric("n₀ · p₀",       f"{n0_t3*p0_t3:.2e} cm⁻⁶" if n0_t3 > 0 and p0_t3 > 0 else "~0")
    c4.metric("n₀p₀ / nᵢ²",    f"{ratio:.5f}")

    if abs(ratio - 1) < 0.02:
        st.success("n₀p₀ = nᵢ²  ✓  (mass action law verified)")
    else:
        st.info(
            f"n₀p₀ / nᵢ² = {ratio:.4f}  "
            "(Boltzmann approximation may be breaking down near the band edges)"
        )

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Doping & Fermi Level
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Doping & Fermi Level Position")
    st.latex(r"""
        n_0 - p_0 = N_d^+ - N_a^- \quad \text{(charge neutrality)}, \qquad
        n_0 p_0 = n_i^2 \quad \text{(mass action law)}
    """)

    col_ctrl, col_right = st.columns([1, 2])

    with col_ctrl:
        doping_type = st.radio(
            "Doping", ["Intrinsic", "n-type  (donors N\u2099)", "p-type  (acceptors N\u2090)"]
        )

        if "n-type" in doping_type:
            Nd_exp = st.slider("log₁₀(N\u2099 / cm⁻³)", 10.0, 20.0, 15.0, step=0.05)
            Nd, Na = 10**Nd_exp, 0.0
            st.caption(f"Nᵈ = {Nd:.2e} cm⁻³   (nᵢ = {ni_val:.2e})")
        elif "p-type" in doping_type:
            Na_exp = st.slider("log₁₀(N\u2090 / cm⁻³)", 10.0, 20.0, 15.0, step=0.05)
            Na, Nd = 10**Na_exp, 0.0
            st.caption(f"Nₐ = {Na:.2e} cm⁻³   (nᵢ = {ni_val:.2e})")
        else:
            Nd = Na = 0.0

        # Compute n₀, p₀, EF all in log-space — no clamping
        if Nd == 0 and Na == 0:
            # Intrinsic: use log_ni directly to avoid log(0) when ni underflows
            log_n0_d = log_ni_v
            log_p0_d = log_ni_v
        else:
            n0_d, p0_d = solve_doping(Nd, Na, ni_val)
            log_n0_d = np.log(n0_d) if n0_d > 0 else log_ni_v
            log_p0_d = np.log(p0_d) if p0_d > 0 else log_ni_v

        if log_n0_d >= log_p0_d:
            EF_d = EF_from_log_n(log_n0_d, Nc, Ec, T)
        else:
            EF_d = EF_from_log_p(log_p0_d, Nv, Ev, T)
        EF_d = np.clip(EF_d, Ev - 0.3, Ec + 0.3)

        n0_d_val = float(np.exp(log_n0_d)) if log_n0_d > -700 else 0.0
        p0_d_val = float(np.exp(log_p0_d)) if log_p0_d > -700 else 0.0
        log_ratio_d = log_n0_d + log_p0_d - 2.0 * log_ni_v
        ratio_d = float(np.exp(np.clip(log_ratio_d, -30, 30)))

        st.divider()
        st.markdown("**Results**")
        st.markdown(f"""
| Quantity | Value |
|:--|--:|
| n<sub>0</sub> | {fmt_density(n0_d_val)} |
| p<sub>0</sub> | {fmt_density(p0_d_val)} |
| E<sub>F</sub> | {EF_d:.4f} eV |
| E<sub>F</sub> − E<sub>i</sub> | {(EF_d - Ei)*1e3:.1f} meV |
| n<sub>0</sub>p<sub>0</sub> | {n0_d_val*p0_d_val:.2e} cm⁻⁶ |
| n<sub>i</sub>² | {ni_val**2:.2e} cm⁻⁶ |
""", unsafe_allow_html=True)

        if abs(ratio_d - 1) < 0.02:
            st.success("n₀p₀ ≈ nᵢ²  ✓")
        else:
            st.warning(f"n₀p₀ / nᵢ² = {ratio_d:.4f}")

    with col_right:
        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 6))

        band_diagram_style(axes2[0], Ev, Ec, EF_d, Ei,
                           title=f"{material_name}  |  T = {T} K")
        if abs(EF_d - Ei) > 0.003:
            axes2[0].annotate(
                "", xy=(0.5, EF_d), xytext=(0.5, Ei),
                arrowprops=dict(arrowstyle="<->", color="purple", lw=1.5),
            )
            axes2[0].text(0.52, (EF_d + Ei) / 2, f"{(EF_d-Ei)*1e3:+.0f} meV",
                          color="purple", fontsize=9, va="center")

        # EF vs doping sweep — all in log-space, no clamping
        N_sweep = np.logspace(10, 20, 400)
        EF_sweep = []
        for Ndop in N_sweep:
            if "n-type" in doping_type:
                ns, ps = solve_doping(Ndop, 0.0, ni_val)
                ln = np.log(ns) if ns > 0 else log_ni_v
                lp = np.log(ps) if ps > 0 else log_ni_v
            elif "p-type" in doping_type:
                ns, ps = solve_doping(0.0, Ndop, ni_val)
                ln = np.log(ns) if ns > 0 else log_ni_v
                lp = np.log(ps) if ps > 0 else log_ni_v
            else:
                ln = lp = log_ni_v
            ef = EF_from_log_n(ln, Nc, Ec, T) if ln >= lp else EF_from_log_p(lp, Nv, Ev, T)
            EF_sweep.append(np.clip(ef, Ev - 0.35, Ec + 0.35))

        ax_sweep = axes2[1]
        ax_sweep.semilogx(N_sweep, EF_sweep, "g-", lw=2, label="$E_F$")
        ax_sweep.axhline(Ec, color="red",  lw=1.2, ls="--", label="$E_c$")
        ax_sweep.axhline(Ev, color="blue", lw=1.2, ls="--", label="$E_v$")
        ax_sweep.axhline(Ei, color="gray", lw=1,   ls=":",  label="$E_i$")

        if "n-type" in doping_type:
            ax_sweep.axvline(Nd, color="orange", lw=1.5, ls="-.", label=f"$N_d$={Nd:.1e}")
            ax_sweep.scatter([Nd], [EF_d], color="orange", zorder=5, s=60)
        elif "p-type" in doping_type:
            ax_sweep.axvline(Na, color="orange", lw=1.5, ls="-.", label=f"$N_a$={Na:.1e}")
            ax_sweep.scatter([Na], [EF_d], color="orange", zorder=5, s=60)

        ax_sweep.axhspan(Ev, Ec, alpha=0.05, color="yellow")
        ax_sweep.set_xlabel("Doping concentration (cm⁻³)", fontsize=10)
        ax_sweep.set_ylabel("$E_F$ (eV)", fontsize=10)
        label_ax = ("$N_d$" if "n-type" in doping_type
                    else ("$N_a$" if "p-type" in doping_type else "N"))
        ax_sweep.set_title(f"$E_F$ vs. {label_ax}", fontsize=10)
        ax_sweep.legend(fontsize=8)
        ax_sweep.grid(alpha=0.3)
        ax_sweep.set_ylim(Ev - 0.35, Ec + 0.35)

        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        # n₀ and p₀ vs doping — no clamping; use np.maximum only for log-plot rendering
        st.subheader("Carrier densities vs. doping")
        n_sweep, p_sweep = [], []
        for Ndop in N_sweep:
            if "n-type" in doping_type:
                ns, ps = solve_doping(Ndop, 0.0, ni_val)
            elif "p-type" in doping_type:
                ns, ps = solve_doping(0.0, Ndop, ni_val)
            else:
                ns = ps = ni_val
            n_sweep.append(ns if ns > 0 else ni_val)
            p_sweep.append(ps if ps > 0 else ni_val)

        n_arr = np.array(n_sweep)
        p_arr = np.array(p_sweep)
        plot_floor = max(ni_val * 1e-3, 1e-40)  # avoid log(0) on plot only

        fig3, ax3 = plt.subplots(figsize=(8, 3.5))
        ax3.loglog(N_sweep, np.maximum(n_arr, plot_floor), "r-",  lw=2, label="$n_0$ (electrons)")
        ax3.loglog(N_sweep, np.maximum(p_arr, plot_floor), "b--", lw=2, label="$p_0$ (holes)")
        if ni_val > 0:
            ax3.axhline(ni_val, color="gray", lw=1.2, ls=":", label=f"$n_i$ = {ni_val:.1e}")
        if "n-type" in doping_type and Nd > 0:
            ax3.axvline(Nd, color="orange", lw=1.5, ls="-.", label=f"$N_d$ = {Nd:.1e}")
        elif "p-type" in doping_type and Na > 0:
            ax3.axvline(Na, color="orange", lw=1.5, ls="-.", label=f"$N_a$ = {Na:.1e}")
        ax3.set_xlabel("Doping concentration (cm⁻³)", fontsize=10)
        ax3.set_ylabel("Carrier density (cm⁻³)", fontsize=10)
        ax3.set_title("$n_0$ and $p_0$  —  note: $n_0 p_0 = n_i^2$ always", fontsize=10)
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3, which="both")
        st.pyplot(fig3, use_container_width=True)
        plt.close()
