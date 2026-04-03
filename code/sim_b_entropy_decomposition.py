"""
sim_b_entropy_decomposition.py

Show that apparent entropy decrease in a low-dimensional projection
(dS_d < 0) coincides with entropy increase in the complementary
conditional entropy (dS_{D-d|d} > 0), preserving total entropy.

Uses 5 coupled Rossler oscillators (D=15). The exact Gaussian
decomposition S_D = S_d + S_{D-d|d} guarantees that when total
entropy is approximately stable, drops in S_d must be compensated
by rises in S_{D-d|d}.

The bottom panel normalises both entropy changes by their respective
standard deviations, making the anti-correlated structure visible even
though S_d fluctuates on a smaller absolute scale than S_{D-d|d}.

Generates: ../figures/fig3_entropy_decomposition.pdf
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Rossler system
# ---------------------------------------------------------------------------

ROSSLER_A = 0.2
ROSSLER_B = 0.2
ROSSLER_C = 5.7
EPSILON = 0.15


def coupled_rossler_rhs(t, state, n_osc, a, b, c, eps):
    """Right-hand side for N diffusively coupled Rossler oscillators."""
    s = state.reshape(n_osc, 3)
    ds = np.zeros_like(s)
    x_mean = s[:, 0].mean()
    for i in range(n_osc):
        xi, yi, zi = s[i]
        a_i = a + 0.03 * (i - n_osc / 2.0)
        ds[i, 0] = -yi - zi + eps * (x_mean - xi)
        ds[i, 1] = xi + a_i * yi
        ds[i, 2] = b + zi * (xi - c)
    return ds.ravel()


def integrate_rossler(n_osc, t_total=1500.0, dt_save=0.05, t_transient=300.0):
    """Integrate coupled Rossler system."""
    D = 3 * n_osc
    np.random.seed(42)
    y0 = np.random.randn(D) * 1.0
    t_eval = np.arange(t_transient, t_total, dt_save)
    sol = solve_ivp(
        coupled_rossler_rhs, (0.0, t_total), y0,
        args=(n_osc, ROSSLER_A, ROSSLER_B, ROSSLER_C, EPSILON),
        method="RK45", t_eval=t_eval, rtol=1e-8, atol=1e-10, max_step=0.05,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return sol.t, sol.y.T


# ---------------------------------------------------------------------------
# Exact Gaussian entropy decomposition
# ---------------------------------------------------------------------------


def entropy_decomposition(data, d_obs, reg=1e-10):
    """Compute S_total, S_obs, S_cond where S_total = S_obs + S_cond."""
    n, D = data.shape
    if n < D + 2:
        return np.nan, np.nan, np.nan

    cov = np.cov(data, rowvar=False)
    cov += reg * np.eye(D)
    d_hid = D - d_obs

    Sigma_oo = cov[:d_obs, :d_obs]
    Sigma_hh = cov[d_obs:, d_obs:]
    Sigma_ho = cov[d_obs:, :d_obs]
    Sigma_oh = cov[:d_obs, d_obs:]

    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return np.nan, np.nan, np.nan
    S_total = 0.5 * logdet

    sign_o, logdet_o = np.linalg.slogdet(Sigma_oo)
    if sign_o <= 0:
        return np.nan, np.nan, np.nan
    S_obs = 0.5 * logdet_o

    try:
        Sigma_oo_inv = np.linalg.inv(Sigma_oo)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan
    Sigma_cond = Sigma_hh - Sigma_ho @ Sigma_oo_inv @ Sigma_oh
    Sigma_cond += reg * np.eye(d_hid)

    sign_c, logdet_c = np.linalg.slogdet(Sigma_cond)
    if sign_c <= 0:
        return np.nan, np.nan, np.nan
    S_cond = 0.5 * logdet_c

    return S_total, S_obs, S_cond


def sliding_window_decomposition(traj, d_obs, window_size, step_size):
    """Compute windowed exact entropy decomposition."""
    n_points = len(traj)
    t_centres, S_tot_list, S_obs_list, S_cond_list = [], [], [], []

    for start in range(0, n_points - window_size + 1, step_size):
        end = start + window_size
        s_tot, s_obs, s_cond = entropy_decomposition(traj[start:end], d_obs)
        if np.isnan(s_tot):
            continue
        t_centres.append((start + end) / 2.0)
        S_tot_list.append(s_tot)
        S_obs_list.append(s_obs)
        S_cond_list.append(s_cond)

    return {
        "t_centres": np.array(t_centres),
        "S_total": np.array(S_tot_list),
        "S_obs": np.array(S_obs_list),
        "S_cond": np.array(S_cond_list),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_simulation():
    """Run the entropy decomposition analysis."""
    n_osc = 5
    D = 3 * n_osc
    d_obs = 6  # first two oscillators

    print(f"Integrating {n_osc} coupled Rossler oscillators (D={D})...")
    t_arr, traj = integrate_rossler(n_osc)
    print(f"  Trajectory: {traj.shape[0]} points, {traj.shape[1]} dimensions")
    dt = t_arr[1] - t_arr[0]

    window_time = 18.0
    window_size = int(window_time / dt)
    step_time = 3.0
    step_size = int(step_time / dt)

    print(f"  Window: {window_size} samples ({window_time:.1f} time units)")
    print(f"  Step: {step_size} samples ({step_time:.1f} time units)")

    ent = sliding_window_decomposition(traj, d_obs, window_size, step_size)
    n_w = len(ent["t_centres"])
    print(f"  Windows: {n_w}")

    residual = np.abs(ent["S_total"] - ent["S_obs"] - ent["S_cond"])
    print(f"  Decomposition error: max={residual.max():.2e}, mean={residual.mean():.2e}")

    t_time = ent["t_centres"] * dt + t_arr[0]

    # Finite differences
    dS_obs = np.diff(ent["S_obs"])
    dS_cond = np.diff(ent["S_cond"])
    dS_tot = np.diff(ent["S_total"])
    t_diff = 0.5 * (t_time[:-1] + t_time[1:])

    # Smooth
    def smooth(x, w=5):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    pad = 2
    dS_obs_s = smooth(dS_obs)
    dS_cond_s = smooth(dS_cond)
    dS_tot_s = smooth(dS_tot)
    t_diff_s = t_diff[pad:pad + len(dS_obs_s)]

    corr = np.corrcoef(dS_obs_s, dS_cond_s)[0, 1]
    neg_mask = dS_obs_s < 0
    n_neg = np.sum(neg_mask)
    n_comp = np.sum(neg_mask & (dS_cond_s > 0))
    print(f"  rho(dS_d, dS_{{D-d|d}}) = {corr:.3f}")
    print(f"  dS_d < 0: {n_neg}/{len(neg_mask)}")
    print(f"  Compensated: {n_comp}/{n_neg}")

    return {
        "t_time": t_time,
        "S_obs": ent["S_obs"],
        "S_cond": ent["S_cond"],
        "S_total": ent["S_total"],
        "t_diff": t_diff_s,
        "dS_obs": dS_obs_s,
        "dS_cond": dS_cond_s,
        "dS_total": dS_tot_s,
        "D": D,
        "d_obs": d_obs,
    }


def make_figure(data):
    """Create the two-panel entropy decomposition figure."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.12},
    )

    t_time = data["t_time"]
    t_diff = data["t_diff"]
    D = data["D"]
    d = data["d_obs"]

    c_obs = "#2166ac"
    c_cond = "#b2182b"
    c_tot = "#1a1a1a"

    # ---- Top panel: entropy levels ----
    ax1.plot(t_time, data["S_obs"], color=c_obs, linewidth=1.2,
             label=rf"$S_d$ (observed, $d={d}$)")
    ax1.plot(t_time, data["S_cond"], color=c_cond, linewidth=1.2,
             label=rf"$S_{{D-d \mid d}}$ (hidden$\mid$obs, ${D-d}$ dims)")
    ax1.plot(t_time, data["S_total"], color=c_tot, linewidth=1.5,
             label=rf"$S_D$ (total, $D={D}$)")

    ax1.set_ylabel("Differential entropy (nats)", fontsize=11)
    ax1.legend(fontsize=9, frameon=False, loc="lower left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(labelsize=9)
    ax1.set_title(
        r"Exact decomposition: $S_D = S_d + S_{D-d \mid d}$",
        fontsize=12,
    )

    # ---- Bottom panel: NORMALISED entropy changes ----
    # Normalise each series by its own std so both are visually comparable
    dS_obs_n = data["dS_obs"] / (np.std(data["dS_obs"]) + 1e-15)
    dS_cond_n = data["dS_cond"] / (np.std(data["dS_cond"]) + 1e-15)

    ax2.axhline(0, color="#999999", linewidth=0.5, zorder=0)
    ax2.plot(t_diff, dS_obs_n, color=c_obs, linewidth=1.0, alpha=0.9)
    ax2.plot(t_diff, dS_cond_n, color=c_cond, linewidth=1.0, alpha=0.9)

    # Shade where dS_obs < 0
    neg_mask = data["dS_obs"] < 0
    changes = np.diff(neg_mask.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if neg_mask[0]:
        starts = np.concatenate([[0], starts])
    if neg_mask[-1]:
        ends = np.concatenate([ends, [len(neg_mask)]])

    for s, e in zip(starts, ends):
        e_idx = min(e, len(t_diff) - 1)
        ax2.axvspan(t_diff[s], t_diff[e_idx], alpha=0.10, color=c_obs, zorder=0)

    ax2.set_xlabel("Time (Rossler units)", fontsize=11)
    ax2.set_ylabel(r"$\Delta S / \sigma_{\Delta S}$ (standardised)", fontsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(labelsize=9)

    leg_elements = [
        plt.Line2D([0], [0], color=c_obs, linewidth=1.2,
                    label=rf"$\Delta S_d$ (observed)"),
        plt.Line2D([0], [0], color=c_cond, linewidth=1.2,
                    label=rf"$\Delta S_{{D-d \mid d}}$ (hidden$\mid$obs)"),
        mpatches.Patch(facecolor=c_obs, alpha=0.12,
                       label=r"$\Delta S_d < 0$ (apparent negentropy)"),
    ]
    ax2.legend(handles=leg_elements, fontsize=9, frameon=False,
               loc="upper right")

    # Stats
    n_neg = np.sum(neg_mask)
    n_total = len(neg_mask)
    n_comp = np.sum(neg_mask & (data["dS_cond"] > 0))
    pct_neg = 100 * n_neg / n_total
    pct_comp = 100 * n_comp / max(1, n_neg)
    corr = np.corrcoef(data["dS_obs"], data["dS_cond"])[0, 1]

    # Stability of total: ratio of std(dS_D) to std(dS_d)
    ratio_std = np.std(data["dS_total"]) / (np.std(data["dS_obs"]) + 1e-15)

    summary = (
        f"$\\rho(\\Delta S_d,\\, \\Delta S_{{D-d|d}})$ = {corr:.2f}\n"
        f"Windows $\\Delta S_d < 0$: {n_neg}/{n_total} ({pct_neg:.0f}%)\n"
        f"Compensated ($\\Delta S_{{D-d|d}} > 0$): "
        f"{n_comp}/{n_neg} ({pct_comp:.0f}%)\n"
        f"$\\sigma_{{\\Delta S_D}} / \\sigma_{{\\Delta S_d}}$ = {ratio_std:.1f} "
        f"(total vs observed volatility)"
    )
    ax2.text(
        0.02, 0.04, summary,
        transform=ax2.transAxes, fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    outpath = "../figures/fig3_entropy_decomposition.pdf"
    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    print(f"\nFigure saved to {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    np.random.seed(42)
    data = run_simulation()
    make_figure(data)
