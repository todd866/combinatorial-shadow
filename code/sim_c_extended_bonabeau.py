"""
Extended Bonabeau hierarchy model with tunable kernel dimensionality.

Extends Bonabeau's (1996) hierarchy emergence model to agents with D-dimensional
state vectors. Shows that hierarchy structure complexity tracks kernel
dimensionality: higher D yields lower rank consistency across projections,
higher effective dimensionality of social structure, and reduced cross-axis
transitivity.

Key insight: in the 1-D model every projection sees the same ranking, so the
hierarchy is perfectly linear and fully transitive.  As D grows, different
projection axes reveal *different* rankings — the hierarchy becomes a partial
order whose cross-projection rank consistency drops and whose cross-projection
transitivity decreases.

Usage:
    python sim_c_extended_bonabeau.py

Output:
    ../figures/fig5_hierarchy_vs_D.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
N_AGENTS = 50
T_ROUNDS = 10_000
DELTA = 1.0
DECAY = 0.001               # per-round multiplicative decay toward zero
NOISE_SCALE = 1e-6           # tiebreak noise
D_VALUES = [1, 2, 4, 8, 16, 32]
N_REPLICATES = 20
N_PROJECTION_AXES = 200      # random axes for measuring projected hierarchy
SEED = 42


# ---------------------------------------------------------------------------
# Hierarchy metrics
# ---------------------------------------------------------------------------

def rank_consistency(states: np.ndarray, rng,
                     n_axes: int = 200) -> float:
    """Mean pairwise Spearman rank correlation across random projections.

    For D=1 all projections yield +-same ranking (|rho|~1).
    For high D, different projections yield different rankings (|rho| -> 0).

    Returns mean |rho| across sampled pairs of projection axes.
    """
    n, d = states.shape
    axes = rng.randn(n_axes, d)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True) + 1e-12
    proj = states @ axes.T  # (N, n_axes)

    # Compute ranks for each axis
    ranks = np.zeros_like(proj)
    for col in range(n_axes):
        order = np.argsort(proj[:, col])
        ranks[order, col] = np.arange(n)

    # Sample pairs of axes and compute |Spearman rho|
    n_pairs = min(1000, n_axes * (n_axes - 1) // 2)
    abs_rhos = []
    for _ in range(n_pairs):
        a1, a2 = rng.choice(n_axes, size=2, replace=False)
        r1, r2 = ranks[:, a1], ranks[:, a2]
        d_sq = np.sum((r1 - r2) ** 2)
        rho = 1 - 6 * d_sq / (n * (n ** 2 - 1))
        abs_rhos.append(abs(rho))

    return np.mean(abs_rhos)


def stochastic_dominance_transitivity(states: np.ndarray, rng,
                                      n_axes: int = 500) -> float:
    """Transitivity of the stochastic dominance relation.

    Define the dominance matrix P[i,j] = fraction of random projection axes
    on which agent i outscores agent j.  Then agent i dominates j iff
    P[i,j] > 0.5.

    Transitivity = fraction of ordered triples (i,j,k) where:
      P[i,j] > 0.5 AND P[j,k] > 0.5 implies P[i,k] > 0.5.

    For D=1: all axes give the same ranking (up to sign), so P[i,j] is
    always 0 or 1 and the dominance relation is perfectly transitive.
    For high D: axes are quasi-orthogonal, P[i,j] ~ 0.5 for most pairs,
    and the dominance relation becomes intransitive.
    """
    n, d = states.shape
    if n < 3:
        return 1.0

    # Generate random projection axes
    axes = rng.randn(n_axes, d)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True) + 1e-12
    proj = states @ axes.T  # (N, n_axes)

    # Vectorized dominance matrix: P[i,j] = fraction of axes where i > j
    # proj[:, None, :] > proj[None, :, :] gives (N, N, n_axes) bool
    P = np.mean(proj[:, None, :] > proj[None, :, :], axis=2)  # (N, N)

    # Dominance matrix: i dominates j iff P[i,j] > 0.5
    dom = P > 0.5  # (N, N) boolean
    np.fill_diagonal(dom, False)

    # Vectorized transitivity: for all (i,j,k) where dom[i,j] and dom[j,k],
    # check dom[i,k].
    # dom[i,j] AND dom[j,k] is the matrix product dom @ dom (counts paths)
    # dom[i,k] AND (dom @ dom)[i,k] > 0 gives transitive triples
    dom_float = dom.astype(float)
    paths = dom_float @ dom_float  # paths[i,k] = number of j s.t. dom[i,j] AND dom[j,k]
    np.fill_diagonal(paths, 0)

    valid = paths.sum()
    # For each (i,k) with paths[i,k] > 0 and dom[i,k], all paths[i,k] triples
    # are transitive. If not dom[i,k], zero of those triples are transitive.
    transitive = (paths * dom_float).sum()

    if valid == 0:
        return 1.0
    return transitive / valid


def effective_dimensionality(states: np.ndarray) -> float:
    """Participation ratio of the covariance spectrum.

    D_eff = (sum lambda_i)^2 / sum(lambda_i^2)
    """
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
# Extended Bonabeau model
# ---------------------------------------------------------------------------

def run_bonabeau(D: int, rng: np.random.RandomState) -> dict:
    """Run one replicate of the D-dimensional Bonabeau model.

    State update: winner gains along the contest axis, loser loses along it.
    All states decay slightly each round to prevent unbounded drift.
    No clamp to >= 0: agents can have negative components, which is critical
    for producing axis-dependent rankings in D > 1.

    Returns
    -------
    dict with keys: 'states', 'rank_consistency', 'd_eff', 'transitivity'
    """
    states = np.zeros((N_AGENTS, D))

    for _ in range(T_ROUNDS):
        # Global decay — prevents runaway norms, keeps dynamics in steady state
        states *= (1 - DECAY)

        # Pick random pair
        pair = rng.choice(N_AGENTS, size=2, replace=False)
        a, b = pair

        # Random contest axis
        u = rng.randn(D)
        u /= np.linalg.norm(u) + 1e-12

        # Project scores onto contest axis
        score_a = np.dot(states[a], u) + rng.normal(0, NOISE_SCALE)
        score_b = np.dot(states[b], u) + rng.normal(0, NOISE_SCALE)

        if score_a >= score_b:
            winner, loser = a, b
        else:
            winner, loser = b, a

        states[winner] += DELTA * u
        states[loser] -= DELTA * u

    # --- Metrics ---

    # Rank consistency across random projections
    rc = rank_consistency(states, rng, n_axes=N_PROJECTION_AXES)

    # Effective dimensionality of social structure
    d_eff = effective_dimensionality(states)

    # Stochastic dominance transitivity (use more axes for stable estimate)
    trans = stochastic_dominance_transitivity(states, rng, n_axes=500)

    return {
        'states': states,
        'rank_consistency': rc,
        'd_eff': d_eff,
        'transitivity': trans,
    }


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

def run_experiment():
    """Run all D values x replicates and collect results."""
    results = {d: {'rank_consistency': [], 'd_eff': [], 'transitivity': []}
               for d in D_VALUES}

    total = len(D_VALUES) * N_REPLICATES
    count = 0

    for D in D_VALUES:
        for rep in range(N_REPLICATES):
            rng = np.random.RandomState(SEED + D * 1000 + rep)

            out = run_bonabeau(D, rng)
            results[D]['rank_consistency'].append(out['rank_consistency'])
            results[D]['d_eff'].append(out['d_eff'])
            results[D]['transitivity'].append(out['transitivity'])

            count += 1
            if count % 5 == 0 or count == total:
                print(f"  [{count}/{total}] D={D:>2}, rep={rep+1:>2}  "
                      f"|rho|={out['rank_consistency']:.3f}  "
                      f"D_eff={out['d_eff']:.2f}  "
                      f"trans={out['transitivity']:.3f}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_figure(results: dict, outpath: str):
    """Three-panel figure: rank consistency, D_eff, transitivity vs kernel D."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    D_arr = np.array(D_VALUES)

    # Collect means and stds
    rc_mean = np.array([np.mean(results[d]['rank_consistency'])
                        for d in D_VALUES])
    rc_std = np.array([np.std(results[d]['rank_consistency'])
                       for d in D_VALUES])
    deff_mean = np.array([np.mean(results[d]['d_eff']) for d in D_VALUES])
    deff_std = np.array([np.std(results[d]['d_eff']) for d in D_VALUES])
    trans_mean = np.array([np.mean(results[d]['transitivity'])
                           for d in D_VALUES])
    trans_std = np.array([np.std(results[d]['transitivity'])
                          for d in D_VALUES])

    # Common style
    marker_kw = dict(marker='o', markersize=5, capsize=3, linewidth=1.2,
                     color='#2c3e50', ecolor='#7f8c8d')

    # --- Panel A: Rank consistency ---
    ax = axes[0]
    ax.errorbar(D_arr, rc_mean, yerr=rc_std, **marker_kw)
    ax.set_xlabel(r'Kernel dimension $D$', fontsize=11)
    ax.set_ylabel(r'Rank consistency $\langle|\rho|\rangle$', fontsize=11)
    ax.set_title('A', fontsize=13, fontweight='bold', loc='left')
    ax.set_xscale('log', base=2)
    ax.set_xticks(D_VALUES)
    ax.set_xticklabels([str(d) for d in D_VALUES])
    ax.set_ylim(-0.05, 1.05)

    # --- Panel B: D_eff ---
    ax = axes[1]
    ax.errorbar(D_arr, deff_mean, yerr=deff_std, **marker_kw)
    ax.plot(D_arr, D_arr, '--', color='#e74c3c', linewidth=1, alpha=0.7,
            label=r'$D_{\mathrm{eff}} = D$')
    ax.set_xlabel(r'Kernel dimension $D$', fontsize=11)
    ax.set_ylabel(r'Social structure $D_{\mathrm{eff}}$', fontsize=11)
    ax.set_title('B', fontsize=13, fontweight='bold', loc='left')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xticks(D_VALUES)
    ax.set_xticklabels([str(d) for d in D_VALUES])
    ax.legend(fontsize=9, frameon=False)
    ax.set_ylim(bottom=0.5)

    # --- Panel C: Cross-projection transitivity ---
    ax = axes[2]
    ax.errorbar(D_arr, trans_mean, yerr=trans_std, **marker_kw)
    ax.axhline(0.5, ls=':', color='#95a5a6', linewidth=0.8,
               label='chance')
    ax.set_xlabel(r'Kernel dimension $D$', fontsize=11)
    ax.set_ylabel('Dominance transitivity', fontsize=11)
    ax.set_title('C', fontsize=13, fontweight='bold', loc='left')
    ax.set_xscale('log', base=2)
    ax.set_xticks(D_VALUES)
    ax.set_xticklabels([str(d) for d in D_VALUES])
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=9, frameon=False, loc='upper right')

    # Global style
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
    print("Extended Bonabeau model: hierarchy vs kernel dimensionality")
    print(f"  N={N_AGENTS}, T={T_ROUNDS}, delta={DELTA}, decay={DECAY}")
    print(f"  D values: {D_VALUES}")
    print(f"  Replicates per D: {N_REPLICATES}")
    print()

    results = run_experiment()

    outpath = os.path.join(os.path.dirname(__file__),
                           '..', 'figures', 'fig5_hierarchy_vs_D.pdf')
    outpath = os.path.normpath(outpath)
    make_figure(results, outpath)

    # Print summary table
    print("\nSummary (mean +/- std):")
    print(f"{'D':>4}  {'|rho| consist.':>14}  {'D_eff':>14}  "
          f"{'Transitivity':>14}")
    print("-" * 54)
    for d in D_VALUES:
        rm = np.mean(results[d]['rank_consistency'])
        rs = np.std(results[d]['rank_consistency'])
        dm = np.mean(results[d]['d_eff'])
        ds = np.std(results[d]['d_eff'])
        tm = np.mean(results[d]['transitivity'])
        ts = np.std(results[d]['transitivity'])
        print(f"{d:>4}  {rm:>6.3f} +/- {rs:<5.3f}  "
              f"{dm:>6.2f} +/- {ds:<5.2f}  "
              f"{tm:>6.3f} +/- {ts:<5.3f}")
