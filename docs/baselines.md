# Baseline Algorithms (Phase 1 & 2)

We use **Stable-Baselines3** to ensure robust, paper-faithful implementations of standard model-free algorithms. These act as our gold standard for measuring custom approaches later on.

All baseline scripts are found in `baselines/`.

## Proximal Policy Optimization (PPO)
*File: `baselines/ppo_experiment.py`*

**Characteristics:**
- On-policy
- Model-free
- Discrete or Continuous action spaces
- Highly stable due to clipped surrogate objective function that prevents destructively large policy updates.

**Tested On:**
- `CartPole-v1`
- `LunarLander-v3`

## Deep Q-Network (DQN)
*File: `baselines/dqn_experiment.py`*

**Characteristics:**
- Off-policy
- Model-free
- Discrete action spaces only
- Bootstraps value estimates and utilizes a target network + replay buffer to stabilize Q-learning.

**Tested On:**
- `CartPole-v1`
- `LunarLander-v3`

## Double DQN
*File: `baselines/double_dqn_experiment.py`*

**Paper alignment (van Hasselt et al., 2015):**
- Decouples action selection and action evaluation in TD target:
  - `a* = argmax_a Q_online(s', a)`
  - `y = r + γ Q_target(s', a*)`
- Reduces overestimation bias compared to vanilla DQN.

**Tested On:**
- `CartPole-v1`
- `LunarLander-v3`

## Prioritized Experience Replay DQN (PER-DQN)
*File: `baselines/per_dqn_experiment.py`*

**Paper alignment (Schaul et al., 2015):**
- Proportional prioritization: `P(i) ∝ |δ_i|^α`
- Importance-sampling correction weights with annealed `β`
- Priority updates after each gradient step from latest TD errors.

**Tested On:**
- `CartPole-v1`
- `LunarLander-v3`

## Rainbow DQN
*File: `baselines/rainbow_experiment.py`*

**Paper alignment (Hessel et al., 2018):**
- Combined components:
  - Double DQN
  - Dueling network
  - Prioritized replay
  - Multi-step returns (`n=3`)
  - Noisy linear exploration
  - Distributional RL (C51 with 51 atoms)
- Adapted to low-dimensional state inputs via MLP encoder instead of Atari CNN.

**Tested On:**
- `CartPole-v1`
- `LunarLander-v3`

## Soft Actor-Critic (SAC)
*File: `baselines/sac_experiment.py`*

**Characteristics:**
- Off-policy
- Model-free
- Continuous action spaces only
- Entropy-regularized: Maximizes both expected return and policy entropy. This encourages robust exploration and captures broader optimal policies compared to deterministic methods.

**Tested On:**
- `Pendulum-v1`
- `LunarLanderContinuous-v3`

## Variance Analysis (Phase 2)

In `experiments/compare_phase1.py`, we run 3 seeds for each environment/algorithm combo. Because RL can be highly sensitive to initial weights and environment seeds, single-run curves are misleading.

The analysis outputs curves to `analysis/figures/`, plotting the **mean ± shaded standard deviation**. This clearly visualizes, for instance, how much more variance DQN exhibits early in training compared to the stable climb of PPO.
