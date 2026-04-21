# MVE (Model-Based Value Expansion)

## 1. Overview

**Model-Based Value Expansion** (Feinberg et al., 2018) uses a learned dynamics model to **unroll** the value target over several steps, then bootstraps with the value network at the end of the imagined rollouts. This can reduce bias in value targets compared to one-step bootstrapping when the model is accurate.

Implementation: [`train_mve_steve`](../../src/rl_experiments/advanced/mve_steve/mve_steve_agent.py) with `method="mve"`.

---

## 2. Problem setting

Given $s$, approximate rollout under policy $\pi_\theta$ for $h$ steps:


$$
\hat{G}_h(s) = \sum_{t=0}^{h-1} \gamma^t r_t + \gamma^h V_\psi(s_h),
$$


where $s_{t+1} \sim \hat{T}(s_t, a_t)$, $a_t = \pi_\theta(s_t)$.

---

## 3. Intuition

- One-step TD targets can be **high-variance**; multi-step targets can reduce variance **if** the model is good.
- Short horizons control **compounding error**.

---

## 4. Mathematical formulation (this code)

The trainer builds a **model rollout** of length `horizon` (3 for MVE) and uses the terminal value $V(s_h)$ for bootstrap. Value loss is MSE between $V(s)$ and detached target.

---

## 5. Architecture

```mermaid
flowchart LR
  Ens[ensemble_dynamics] --> Roll[rollout_r_sum]
  Roll --> Tail[gamma_h_V_s_h]
  Tail --> Vloss[value_regression]
```

---

## 6. Code anchor

```python
horizon = 3 if method == "mve" else 5
# ...
target = g + tail  # MVE path without ensemble-variance weighting
v_loss = F.mse_loss(v, target.detach())
```

---

## 7. References

1. Feinberg, V., et al. (2018). *Model-Based Value Estimation for Efficient Model-Free Reinforcement Learning.* ICML.

---

## Appendix: Pseudocode and formal notes

Notation: [`00_notation_and_conventions.md`](00_notation_and_conventions.md). $k$-step error: [`theoretical_appendix_model_based.md`](theoretical_appendix_model_based.md).

### A. Pseudocode (model-based value expansion for targets)

```text
Learn model Ĉ and value V_ψ
For each transition (s,a), imagine k steps under model and policy: (s_0,…,s_k)
Bootstrap target:
  Ĝ = Σ_{t=0}^{k−1} γ^t r_t + γ^k V_ψ(s_k)
Train critic toward Ĝ (and train model on real data)
Policy improvement uses improved value estimates (algorithm-specific)
```

### B. Assumptions (informal)

**A1 (horizon $k$).** Expansion depth trades **bias reduction** (closer to $k$-step return) vs **variance** and **model error**.

**A2 (policy coupling).** Rollouts typically use **current** $\pi$; off-policy mismatch affects target validity (importance sampling may be needed in strict analysis).

### C. Remarks

- MVE is a **target augmentation** trick; it does not by itself guarantee **convergence** of actor–critic schemes.
- When the model is poor, **longer** $k$ can **hurt** value estimates—start small.
