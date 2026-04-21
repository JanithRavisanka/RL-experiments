# Theoretical appendix: model-based rollouts and value expansion

This appendix supports the **“next level”** sections on **MBPO**, **PETS**, **MVE**, **STEVE**, and related planners. It states common **structural assumptions** and explains **why short horizons** and **ensembles** appear in implementations.

---

## 1. Compounding error in learned dynamics

Let the true transition be $s_{t+1} = F(s_t,a_t) + \xi_t$ (or stochastic kernel $P$). A learned model $\hat{F}_\phi$ induces rollouts $\hat{s}_{t+1} = \hat{F}_\phi(\hat{s}_t,a_t)$.

**One-step error.** Suppose $\|\hat{F}_\phi(s,a) - F(s,a)\| \le \epsilon$ for $(s,a)$ in a region of interest.

**Multi-step divergence.** Even with Lipschitz dynamics, small per-step errors can accumulate:


$$
\|\hat{s}_k - s_k\| \lesssim \sum_{t=0}^{k-1} L^{k-1-t} \epsilon_t
$$


for Lipschitz constant $L$ (informal). Hence **$k$-step returns** from the model are trusted only for **small $k$** unless additional structure holds.

**Remark.** MBPO explicitly uses **short** model rollouts; MVE/STEVE use **finite** expansion horizons.

---

## 2. Lipschitz-style assumptions (informal)

Many analyses assume:

- **A-Lip:** $\|F(s,a)-F(s',a')\| \le L \|(s,a)-(s',a')\|$ (or expected Lipschitz under stochastic dynamics).
- **Bounded rewards:** $|r(s,a)| \le R_{\max}$.

These yield **bounded** planning error for fixed horizon $H$ when the model is uniformly accurate on the rollout tube.

---

## 3. Ensemble disagreement as epistemic uncertainty

For an ensemble $\{\hat{F}^{(i)}\}_{i=1}^M$, disagreement $\mathrm{Var}_i[\hat{F}^{(i)}]$ proxies **epistemic uncertainty**. STEVE-style weighting **down-weights** targets when rollouts disagree, reducing sensitivity to regions where the model is unreliable.

This is **not** a calibrated Bayesian posterior in general; it is a **heuristic** aligned with ensemble-based uncertainty quantification in deep RL.

---

## 4. Connection to policy improvement

Model-based policy optimization seeks to optimize returns under the **true** MDP using data from $\hat{P}_\phi$. **Monotonic improvement** generally requires either:

- trust regions on policy updates (policy improvement theory), or
- careful mixing of real vs model data (as in MBPO’s analysis), or
- pessimism / penalties when the model is uncertain.

The implementations in this repo are **practical** instantiations; refer to each paper for exact theorems.

---

## 5. References (theory-heavy)

1. Janner, M., Fu, J., Zhang, M., & Levine, S. (2019). *When to Trust Your Model: Model-Based Policy Optimization.* NeurIPS.
2. Feinberg, V., et al. (2018). *Model-Based Value Estimation for Efficient Model-Free Reinforcement Learning.* ICML.
3. Buckman, J., et al. (2018). *Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion.* NeurIPS.
4. Chua, K., et al. (2018). *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models.* NeurIPS.
