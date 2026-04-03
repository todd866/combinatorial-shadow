"""
sim_a_projection_complexity.py

Demonstrate that apparent combinatorial complexity scales with the
dimensional gap (D - d) under projection.

Coupled Rossler oscillators generate high-dimensional chaotic attractors.
Random linear projections to dimension d are analysed for three signatures
of projection-induced complexity:

1. Neighbourhood aliasing: nearby points in the projection that are far
   apart in the full space (non-injectivity of the projection map).
2. Variance captured: fraction of total variance explained by the
   d-dimensional projection (information loss).
3. Conditional entropy: H(X_full | X_proj), the information about the
   full state that is destroyed by projection. This is the "shadow"
   itself -- the hidden combinatorial structure.

Generates: ../figures/fig2_complexity_scaling.pdf
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Rossler system
# ---------------------------------------------------------------------------

ROSSLER_A = 0.2
ROSSLER_B = 0.2
ROSSLER_C = 5.7
EPSILON = 0.05


def coupled_rossler_rhs(t, state, n_osc, a, b, c, eps):
    """Right-hand side for N diffusively coupled Rossler oscillators."""
    s = state.reshape(n_osc, 3)
    ds = np.zeros_like(s)
    x_mean = s[:, 0].mean()
    for i in range(n_osc):
        xi, yi, zi = s[i]
        ds[i, 0] = -yi - zi + eps * (x_mean - xi)
        ds[i, 1] = xi + a * yi
        ds[i, 2] = b + zi * (xi - c)
    return ds.ravel()


def integrate_rossler(n_osc, t_total=600.0, dt_save=0.02, t_transient=200.0):
    """Integrate coupled Rossler system, discard transient, return attractor."""
    D = 3 * n_osc
    np.random.seed(42 + n_osc)
    y0 = np.random.randn(D) * 0.5

    t_eval = np.arange(t_transient, t_total, dt_save)
    sol = solve_ivp(
        coupled_rossler_rhs, (0.0, t_total), y0,
        args=(n_osc, ROSSLER_A, ROSSLER_B, ROSSLER_C, EPSILON),
        method="RK45", t_eval=t_eval, rtol=1e-8, atol=1e-10, max_step=0.05,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed for n_osc={n_osc}: {sol.message}")
    return sol.y.T


# ---------------------------------------------------------------------------
# Complexity measures
# ---------------------------------------------------------------------------


def neighbourhood_aliasing(traj_full, proj, k=20, n_query=2000):
    """Measure how much the projection aliases the full-space structure.

    For a random subset of points, find k nearest neighbours in the
    *projected* space, then compute the mean full-space distance of
    those neighbours. Compare with the mean full-space distance of
    true full-space neighbours.

    Returns the ratio (>=1, where 1 = no aliasing).
    """
    n = len(traj_full)
    idx = np.random.choice(n, min(n_query, n), replace=False)

    tree_proj = cKDTree(proj)
    tree_full = cKDTree(traj_full)

    _, nn_proj = tree_proj.query(proj[idx], k=k + 1)
    nn_proj = nn_proj[:, 1:]

    _, nn_full = tree_full.query(traj_full[idx], k=k + 1)
    nn_full = nn_full[:, 1:]

    full_dists_of_proj_nn = np.zeros(len(idx))
    full_dists_of_full_nn = np.zeros(len(idx))

    for i, qi in enumerate(idx):
        full_dists_of_proj_nn[i] = np.mean(
            np.linalg.norm(traj_full[nn_proj[i]] - traj_full[qi], axis=1)
        )
        full_dists_of_full_nn[i] = np.mean(
            np.linalg.norm(traj_full[nn_full[i]] - traj_full[qi], axis=1)
        )

    ratio = np.mean(full_dists_of_proj_nn) / (np.mean(full_dists_of_full_nn) + 1e-15)
    return ratio


def variance_loss(traj_full, proj):
    """Fraction of total variance NOT captured by the projection."""
    total_var = np.sum(np.var(traj_full, axis=0))
    proj_var = np.sum(np.var(proj, axis=0))
    return 1.0 - proj_var / (total_var + 1e-15)


def conditional_entropy_gaussian(traj_full, proj, reg=1e-10):
    """Estimate H(X_full | X_proj) assuming joint Gaussian.

    Since proj is a deterministic linear function of full,
    H(full, proj) = H(full), so H(full|proj) = H(full) - H(proj).

    Returns total conditional entropy in nats (not per-dimension).
    """
    D = traj_full.shape[1]
    d = proj.shape[1]
    n = len(traj_full)

    # Full-space entropy
    cov_full = np.cov(traj_full, rowvar=False) + reg * np.eye(D)
    _, logdet_full = np.linalg.slogdet(cov_full)

    # Projected-space entropy
    cov_proj = np.cov(proj, rowvar=False)
    if d == 1:
        cov_proj = np.atleast_2d(cov_proj)
    cov_proj += reg * np.eye(d)
    _, logdet_proj = np.linalg.slogdet(cov_proj)

    h_full = 0.5 * logdet_full
    h_proj = 0.5 * logdet_proj
    h_cond = h_full - h_proj  # H(full|proj)

    return h_cond  # total conditional entropy (nats)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def random_orthogonal_projection(traj, d, rng):
    """Project to d dimensions via a random orthonormal basis."""
    D = traj.shape[1]
    P = rng.randn(D, d)
    Q, _ = np.linalg.qr(P)
    return traj @ Q[:, :d]


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def run_simulation():
    """Run the full projection-complexity scaling experiment."""
    osc_counts = [3, 5, 8, 12, 16, 20]
    rng = np.random.RandomState(42)

    results = {}

    for n_osc in osc_counts:
        D = 3 * n_osc
        print(f"Integrating {n_osc} coupled Rossler oscillators (D={D})...")
        traj = integrate_rossler(n_osc)
        print(f"  Trajectory shape: {traj.shape}")

        # Standardise
        traj_std = (traj - traj.mean(axis=0)) / (traj.std(axis=0) + 1e-12)

        # Subsample for speed
        n_pts = min(6000, len(traj_std))
        idx_sub = np.linspace(0, len(traj_std) - 1, n_pts, dtype=int)
        traj_sub = traj_std[idx_sub]

        # Projection dimensions
        d_max = min(D - 1, 40)
        d_values = sorted(set(
            [1, 2, 3, 4, 5]
            + list(np.unique(np.geomspace(2, d_max, 12).astype(int)))
            + [d_max]
        ))
        d_values = [d for d in d_values if 1 <= d <= d_max]

        complexity_by_d = []
        n_proj = 6

        for d in d_values:
            gap = D - d
            print(f"  d={d:2d}  gap={gap:2d} ...")

            alias_vals = []
            vloss_vals = []
            hcond_vals = []

            for _ in range(n_proj):
                proj = random_orthogonal_projection(traj_sub, d, rng)
                alias_vals.append(neighbourhood_aliasing(traj_sub, proj))
                vloss_vals.append(variance_loss(traj_sub, proj))
                hcond_vals.append(conditional_entropy_gaussian(traj_sub, proj))

            complexity_by_d.append({
                "d": d, "D": D, "gap": gap,
                "aliasing": np.mean(alias_vals),
                "var_loss": np.mean(vloss_vals),
                "cond_entropy": np.mean(hcond_vals),
            })

        results[D] = complexity_by_d

    return results


def make_figure(results):
    """Create the three-panel complexity-scaling figure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    measures = [
        ("aliasing", "Neighbourhood aliasing",
         r"Aliasing ratio"),
        ("var_loss", "Variance loss",
         r"$1 - \sigma^2_{\mathrm{proj}} / \sigma^2_{\mathrm{total}}$"),
        ("cond_entropy", "Information destroyed",
         r"$-H(X \mid X_{\mathrm{proj}})$ (nats)"),
    ]

    cmap = plt.cm.viridis
    D_values = sorted(results.keys())
    colours = {
        D: cmap(i / max(1, len(D_values) - 1))
        for i, D in enumerate(D_values)
    }

    for ax, (key, title, ylabel) in zip(axes, measures):
        for D in D_values:
            data = results[D]
            gaps = np.array([r["gap"] for r in data])
            vals = np.array([r[key] for r in data])

            # Negate conditional entropy so "up = more destroyed"
            if key == "cond_entropy":
                vals = -vals

            ax.plot(
                gaps, vals, "o-",
                color=colours[D], markersize=3.5,
                linewidth=1.3, label=f"$D={D}$", alpha=0.85,
            )

            # Linear fit overlay
            finite = np.isfinite(vals)
            if finite.sum() > 3:
                coeffs = np.polyfit(gaps[finite], vals[finite], 1)
                fit_x = np.array([gaps[finite].min(), gaps[finite].max()])
                ax.plot(
                    fit_x, np.polyval(coeffs, fit_x),
                    "--", color=colours[D], alpha=0.35, linewidth=0.8,
                )

        ax.set_xlabel(r"Dimensional gap $(D - d)$", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        ncol=len(D_values), fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    outpath = "../figures/fig2_complexity_scaling.pdf"
    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    print(f"\nFigure saved to {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    np.random.seed(42)
    results = run_simulation()
    make_figure(results)
