# Combinatorial Shadow

**Combinatorial explosion is in the shadow, not the thing.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

A low-dimensional observer encountering a high-dimensional system experiences apparent combinatorial complexity that grows exponentially with the dimensional gap between them. This paper shows that the explosion is a property of the projection, not the system, derives a conditional entropy decomposition clarifying Maxwell's demon as dimensional bookkeeping, and applies these ideas to social hierarchy formation via extended Bonabeau agent models. A speculative coda connects the framework to psychedelic phenomenology.

## Key Results

- **Projection Complexity Scaling** (Theorem 1): The number of system states compatible with any observation scales as ε^{-(δ-d)}, where δ is the attractor dimension and d the observer's
- **Conditional Entropy Decomposition** (Theorem 2): Apparent negentropy in observed dimensions (ΔS_d < 0) requires compensating entropy increase in hidden dimensions (ΔS_{D-d|d} ≥ |ΔS_d|)
- **Hierarchy tracks kernel dimensionality**: D-dimensional agents produce social structures with D_eff ≈ D; low-D kernels produce clean linear hierarchies, high-D kernels produce multi-axis structures
- **Suppression becomes the kernel**: Defence mechanisms (pheromone suppression, ideological compression) that cap dimensionality determine the society's effective dimensionality when strong enough

## Running Simulations

```bash
pip install numpy scipy matplotlib       # Dependencies

# Run individual simulations
python code/sim_a_projection_complexity.py    # Projection complexity scaling (Fig 1)
python code/sim_b_entropy_decomposition.py    # Entropy decomposition (Fig 2)
python code/sim_c_extended_bonabeau.py        # D-dimensional hierarchy model (Fig 3)
python code/sim_d_defence_cost.py             # Defence mechanism cost (Fig 4)
```

Each script is standalone, uses `np.random.seed(42)` for reproducibility, and saves its figure to `figures/`.

## Paper

**Combinatorial Shadow: How Dimensional Mismatch Generates Apparent Complexity, with Applications to Social Hierarchy**

Todd, I. (2026). Preprint.

## Citation

```bibtex
@article{todd2026shadow,
  author  = {Todd, Ian},
  title   = {Combinatorial Shadow: How Dimensional Mismatch Generates Apparent Complexity, with Applications to Social Hierarchy},
  year    = {2026},
  note    = {Preprint}
}
```

## License

MIT License
