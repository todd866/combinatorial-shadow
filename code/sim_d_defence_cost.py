"""
Defence cost simulation: dimensional suppression in high-D hierarchies.

Shows that imposing dimensional caps (defence/suppression mechanisms) on
agents with D_native=16 dimensional state vectors has a cost that grows
with the dimensionality gap.  At a critical cap, the suppression mechanism's
variance dominates the agent-generated variance -- suppression becomes the
dominant attractor of the social system.

The model runs the extended Bonabeau dynamics (random-axis contests with
decay) and periodically projects all agent states onto the top d_cap
principal components, destroying information in the complementary subspace.

Key metrics:
  - Information cost: energy destroyed per suppression event (grows with gap)
  - Rank consistency: how coherent the hierarchy is across projections
    (increases as d_cap shrinks, because suppression forces a low-D structure)
  - D_eff of the social structure: effective dimensionality of agent states
    (tracks d_cap when suppression dominates, tracks D_native when free)

Usage:
    python sim_d_defence_cost.py

Output:
    ../figures/fig6_suppression_cost.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
N_AGENTS = 50
T_ROUNDS = 10_000
DELTA = 1.0
DECAY = 0.001                # per-round multiplicative decay
NOISE_SCALE = 1e-6
D_NATIVE = 16
D_CAP_VALUES = [1, 2, 4, 6, 8, 10, 12, 14, 16]
N_REPLICATES = 20
N_PROJECTION_AXES = 200
SUPPRESSION_INTERVAL = 50    # apply suppression every K rounds
SEED = 42


# ---------------------------------------------------------------------------
# Metrics (same as sim_c)
# ---------------------------------------------------------------------------

def rank_consistency(states: np.ndarray, rng, n_axes: int = 200) -> float:
    """Mean pairwise |Spearman rho| across random projection axes.

    High values mean a single coherent ranking; low values mean the ranking
    depends on which axis you look at.
    """
    n, d = states.shape
    axes = rng.randn(n_axes, d)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True) + 1e-12
    proj = states @ axes.T

    ranks = np.zeros_like(proj)
    for col in range(n_axes):
        order = np.argsort(proj[:, col])
        ranks[order, col] = np.arange(n)

    n_pairs = min(1000, n_axes * (n_axes - 1) // 2)
    abs_rhos = []
    for _ in range(n_pairs):
        a1, a2 = rng.choice(n_axes, size=2, replace=False)
        r1, r2 = ranks[:, a1], ranks[:, a2]
        d_sq = np.sum((r1 - r2) ** 2)
        rho = 1 - 6 * d_sq / (n * (n ** 2 - 1))
        abs_rhos.append(abs(rho))

    return np.mean(abs_rhos)


def effective_dimensionality(states: np.ndarray) -> float:
    """Participation ratio of the covariance spectrum."""
    cov = np.cov(states.T)
    if cov.ndim == 0:
        return 1.0
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0)
    s1 = np.sum(eigvals)
    s2 = np.sum(eigvals ** 2)
    if s2 < 1e-30:
        return 0.0
    return s1 ** 2 / s2


# ---------------------------------------------------------------------------
# Suppression (PCA projection to d_cap dimensions)
# ---------------------------------------------------------------------------

def suppress_to_d_cap(states: np.ndarray, d_cap: int):
    """Project all agent states onto top d_cap principal components.

    Returns
    -------
    projected : array (N, D_native) -- states after projection
    info_cost : float -- sum of squared norms of destroyed components
    """
    if d_cap >= states.shape[1]:
        return states.copy(), 0.0

    mean = np.mean(states, axis=0)
    centered = states - mean

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    basis = Vt[:d_cap]

    coords = centered @ basis.T
    reconstructed = coords @ basis + mean

    residual = states - reconstructed
    info_cost = np.sum(residual ** 2)

    return reconstructed, info_cost


# ---------------------------------------------------------------------------
# Model with suppression
# ---------------------------------------------------------------------------

def run_suppressed_bonabeau(d_cap: int, rng: np.random.RandomState) -> dict:
    """Run Bonabeau model with periodic dimensional suppression.

    Returns dict with: 'avg_info_cost', 'rank_consistency', 'd_eff'
    """
    states = np.zeros((N_AGENTS, D_NATIVE))
    info_costs = []

    for t in range(1, T_ROUNDS + 1):
        # Global decay
        states *= (1 - DECAY)

        # --- Interaction step ---
        pair = rng.choice(N_AGENTS, size=2, replace=False)
        a, b = pair

        u = rng.randn(D_NATIVE)
        u /= np.linalg.norm(u) + 1e-12

        score_a = np.dot(states[a], u) + rng.normal(0, NOISE_SCALE)
        score_b = np.dot(states[b], u) + rng.normal(0, NOISE_SCALE)

        if score_a >= score_b:
            winner, loser = a, b
        else:
            winner, loser = b, a

        states[winner] += DELTA * u
        states[loser] -= DELTA * u

        # --- Periodic suppression ---
        if t % SUPPRESSION_INTERVAL == 0 and d_cap < D_NATIVE:
            states, cost = suppress_to_d_cap(states, d_cap)
            info_costs.append(cost)

    # --- Final metrics ---
    avg_info_cost = np.mean(info_costs) if info_costs else 0.0
    rc = rank_consistency(states, rng, n_axes=N_PROJECTION_AXES)
    d_eff = effective_dimensionality(states)

    return {
        'avg_info_cost': avg_info_cost,
        'rank_consistency': rc,
        'd_eff': d_eff,
    }


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

def run_experiment():
    """Run all d_cap values x replicates."""
    results = {dc: {'info_cost': [], 'rank_consistency': [], 'd_eff': []}
               for dc in D_CAP_VALUES}

    total = len(D_CAP_VALUES) * N_REPLICATES
    count = 0

    for dc in D_CAP_VALUES:
        for rep in range(N_REPLICATES):
            rng = np.random.RandomState(SEED + dc * 1000 + rep)

            out = run_suppressed_bonabeau(dc, rng)
            results[dc]['info_cost'].append(out['avg_info_cost'])
            results[dc]['rank_consistency'].append(out['rank_consistency'])
            results[dc]['d_eff'].append(out['d_eff'])

            count += 1
            if count % 5 == 0 or count == total:
                print(f"  [{count}/{total}] d_cap={dc:>2}, rep={rep+1:>2}  "
                      f"cost={out['avg_info_cost']:>8.1f}  "
                      f"|rho|={out['rank_consistency']:.3f}  "
                      f"D_eff={out['d_eff']:.2f}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_figure(results: dict, outpath: str):
    """Three-panel figure: info cost, rank consistency, D_eff vs d_cap."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    dc_arr = np.array(D_CAP_VALUES)
    gap_arr = D_NATIVE - dc_arr

    # Collect statistics
    cost_mean = np.array([np.mean(results[dc]['info_cost'])
                          for dc in D_CAP_VALUES])
    cost_std = np.array([np.std(results[dc]['info_cost'])
                         for dc in D_CAP_VALUES])
    rc_mean = np.array([np.mean(results[dc]['rank_consistency'])
                        for dc in D_CAP_VALUES])
    rc_std = np.array([np.std(results[dc]['rank_consistency'])
                       for dc in D_CAP_VALUES])
    deff_mean = np.array([np.mean(results[dc]['d_eff'])
                          for dc in D_CAP_VALUES])
    deff_std = np.array([np.std(results[dc]['d_eff'])
                         for dc in D_CAP_VALUES])

    marker_kw = dict(marker='o', markersize=5, capsize=3, linewidth=1.2,
                     color='#2c3e50', ecolor='#7f8c8d')

    # --- Panel A: Information cost vs dimensionality gap ---
    ax = axes[0]
    # Exclude gap=0 (no suppression) from the plot for clarity
    mask = gap_arr > 0
    ax.errorbar(gap_arr[mask], cost_mean[mask], yerr=cost_std[mask],
                **marker_kw)
    ax.set_xlabel(
        r'Dimensionality gap $(D_{\mathrm{native}} - d_{\mathrm{cap}})$',
        fontsize=10)
    ax.set_ylabel('Information cost\n(mean sq. residual per event)',
                  fontsize=10)
    ax.set_title('A', fontsize=13, fontweight='bold', loc='left')
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- Panel B: Rank consistency vs d_cap ---
    ax = axes[1]
    ax.errorbar(dc_arr, rc_mean, yerr=rc_std, **marker_kw)
    ax.set_xlabel(r'Dimensional cap $d_{\mathrm{cap}}$', fontsize=10)
    ax.set_ylabel(r'Rank consistency $\langle|\rho|\rangle$', fontsize=10)
    ax.set_title('B', fontsize=13, fontweight='bold', loc='left')
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- Panel C: D_eff vs d_cap ---
    ax = axes[2]
    ax.errorbar(dc_arr, deff_mean, yerr=deff_std, **marker_kw)
    ax.plot(dc_arr, dc_arr, '--', color='#e74c3c', linewidth=1, alpha=0.7,
            label=r'$D_{\mathrm{eff}} = d_{\mathrm{cap}}$')
    ax.axhline(D_NATIVE, ls=':', color='#95a5a6', linewidth=0.8,
               label=r'$D_{\mathrm{native}}$')
    ax.set_xlabel(r'Dimensional cap $d_{\mathrm{cap}}$', fontsize=10)
    ax.set_ylabel(r'Social structure $D_{\mathrm{eff}}$', fontsize=10)
    ax.set_title('C', fontsize=13, fontweight='bold', loc='left')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, frameon=False, loc='upper left')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=9)

    fig.tight_layout(w_pad=3.0)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {outpath}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Suppression cost model: dimensional caps on high-D hierarchy")
    print(f"  N={N_AGENTS}, T={T_ROUNDS}, D_native={D_NATIVE}, decay={DECAY}")
    print(f"  d_cap values: {D_CAP_VALUES}")
    print(f"  Suppression every {SUPPRESSION_INTERVAL} rounds")
    print(f"  Replicates: {N_REPLICATES}")
    print()

    results = run_experiment()

    outpath = os.path.join(os.path.dirname(__file__),
                           '..', 'figures', 'fig6_suppression_cost.pdf')
    outpath = os.path.normpath(outpath)
    make_figure(results, outpath)

    # Print summary table
    print("\nSummary (mean +/- std):")
    print(f"{'d_cap':>5}  {'gap':>3}  {'Info cost':>16}  "
          f"{'|rho|':>14}  {'D_eff':>14}")
    print("-" * 60)
    for dc in D_CAP_VALUES:
        gap = D_NATIVE - dc
        cm = np.mean(results[dc]['info_cost'])
        cs = np.std(results[dc]['info_cost'])
        rm = np.mean(results[dc]['rank_consistency'])
        rs = np.std(results[dc]['rank_consistency'])
        dm = np.mean(results[dc]['d_eff'])
        ds = np.std(results[dc]['d_eff'])
        print(f"{dc:>5}  {gap:>3}  {cm:>8.1f} +/- {cs:<5.1f}  "
              f"{rm:>6.3f} +/- {rs:<5.3f}  "
              f"{dm:>6.2f} +/- {ds:<5.2f}")
