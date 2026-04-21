# Notation, conventions, and probabilistic setup

This document fixes **notation** used across [`algorithms/index.md`](index.md) so that assumptions and pseudocode in each algorithm appendix are comparable.

---

## 1. Markov decision process

A discounted MDP is a tuple $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$:

| Symbol | Meaning |
|--------|---------|
| $\mathcal{S}$ | State space (measurable; often $\mathbb{R}^d$ in deep RL) |
| $\mathcal{A}$ | Action space (finite for discrete control; Box for continuous) |
| $P(s' \mid s,a)$ | Transition kernel |
| $r(s,a)$ | Reward (may be stochastic; often written $r_t$ for the realized reward) |
| $\gamma \in [0,1)$ | Discount factor |

A **policy** $\pi(a \mid s)$ induces a trajectory distribution; **stationary** state distribution under $\pi$ is denoted $\rho^\pi$ when it exists.

---

## 2. Value functions and advantages


$$
V^\pi(s) = \mathbb{E}_\pi\Big[\sum_{t=0}^\infty \gamma^t r_t \,\Big|\, s_0=s\Big], \quad
Q^\pi(s,a) = \mathbb{E}_\pi\Big[\sum_{t=0}^\infty \gamma^t r_t \,\Big|\, s_0=s, a_0=a\Big].
$$


The **advantage** is $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$. **GAE** provides a family of empirical advantage estimates indexed by $\lambda$.

---

## 3. Function approximation

We write $V_\psi$, $Q_\theta$, $\pi_\phi$ for parametric approximators. **Target networks** use parameters $\bar{\theta}$ updated slowly toward $\theta$. **Bootstrapping** refers to using learned value estimates inside the target for temporal-difference learning.

---

## 4. Model-based rollouts (generic)

Let $\hat{P}_\phi$ be a learned transition model (possibly stochastic). A **$k$-step rollout** from $s_0$ applies


$$
s_{t+1} \sim \hat{P}_\phi(\cdot \mid s_t, a_t), \quad a_t \sim \pi(\cdot \mid s_t).
$$


**Compounding error** refers to the divergence between the distribution of $s_k$ under $\hat{P}_\phi$ versus the true $P$ as $k$ grows; it motivates **short horizons** and **ensembles**.

---

## 5. Stochastic optimization

Unless stated otherwise, expectations $\mathbb{E}[\cdot]$ in objectives are **estimated by minibatch Monte Carlo** over replay buffers or on-policy rollouts. **Unbiasedness** of PER-corrected gradients requires importance weights as in the PER paper; practical implementations use **annealed** $\beta$.

---

## 6. Citations in appendices

Each algorithm page may reference:

- **Assumptions** (A1, A2, …): sufficient regularity for informal statements.
- **Remarks**: qualitative stability, bias–variance trade-offs, or known failure modes.

These are **not** formal theorems proved in this repository; they follow standard RL literature and are stated for **pedagogical rigor**.
