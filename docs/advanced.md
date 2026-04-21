# Advanced Algorithms (Phase 3)

The `advanced/` directory contains custom, from-scratch PyTorch native implementations of cutting edge algorithm architectures. These are designed explicitly to support the PyTorch MPS backend for fast Mac execution.

## Dreamer v3-lite (Model-Based RL)
*Directory: `advanced/dreamer/`*

Dreamer is a prominent model-based paradigm. Instead of learning policies directly from environment interactions, it learns a "World Model" and trains its policy almost entirely inside its own "imagination".

**Architecture Components:**
- **RSSM (Recurrent State Space Model)**: The core latent dynamics model. Contains a deterministic GRU path and a stochastic Gaussian latent path to handle environment uncertainty.
- **World Model**: Combines an Encoder (observations -> latent), the RSSM (dynamics predictive model), and a Decoder (latent -> observations/rewards/continues).
- **Actor-Critic**: The policy (Actor) and value estimator (Critic). These are trained using highly sample-efficient $\lambda$-returns constructed by rolling out the RSSM into the future without touching the real environment.

**How it trains:**
1. Collect episodes from the real environment.
2. Train the World Model to accurately reconstruct inputs and predict the next latents and rewards based on actions.
3. Freeze the World Model. Use the Actor to imagine hundreds of trajectories in the latent space.
4. Train the Actor-Critic based on the imaginary rewards.

## MuZero (Planning-Based RL)
*Directory: `advanced/muzero/`*

MuZero combines the world-model concept with powerful Monte Carlo Tree Search (MCTS). Instead of reconstructing raw observations, MuZero's model only predicts quantities that actually matter for planning: the value, the policy, and the reward.

**Architecture Components:**
- **Representation Network**: Maps real observations ($o_t$) to a hidden state ($s_t$).
- **Dynamics Network**: Maps a hidden state and an action ($s_t, a_t$) to the next hidden state and immediate reward ($s_{t+1}, r_{t+1}$).
- **Prediction Network**: Maps a hidden state ($s_t$) to an estimated value ($v_t$) and policy logits ($p_t$).
- **MCTS**: Uses the learned dynamics network to unroll a search tree. Handles exploration via the UCB (Upper Confidence Bound) formula (PUCT) modified for learned priors.

**How it trains:**
1. Agent plays games against itself (Self-Play) using MCTS equipped with the latest networks. MCTS produces improved policies ($\pi$) and value estimates.
2. Trajectories are stored in a Replay Buffer.
3. During Unroll Training, we sample sequences from the buffer. The network is trained by unrolling the dynamics network over $K$ steps. It is trained to minimize the combined loss of: predicted policies vs MCTS policies, predicted values vs actual $n$-step bootstrapped returns, and predicted rewards vs actual environmental rewards.

## PETS (Probabilistic Ensembles with Trajectory Sampling)
*Directory: `advanced/pets/`*

- Uses an ensemble dynamics model for epistemic uncertainty.
- Plans actions via CEM-based MPC at each decision step.
- Suitable for continuous control environments.

## MBPO (Model-Based Policy Optimization)
*Directory: `advanced/mbpo/`*

- Learns dynamics model alongside a policy (SAC-style backbone).
- Uses short synthetic rollouts to improve sample efficiency.
- Reuses ensemble dynamics infrastructure from PETS-style modeling.

## PlaNet
*Directory: `advanced/planet/`*

- Pixel encoder + latent RSSM transition model.
- Uses planning in latent space instead of direct policy optimization.
- Logs latent/reward objective components for diagnostics.

## TD-MPC / TD-MPC2
*Directory: `advanced/tdmpc/`*

- Latent-model control with short-horizon planning and terminal value backup.
- `tdmpc2` mode increases planning horizon and backup aggressiveness.

## World Models
*Directory: `advanced/world_models/`*

- VAE latent representation learning.
- Recurrent latent dynamics (MDN-RNN style approximation).
- Controller optimization over latent features.

## I2A (Imagination-Augmented Agents)
*Directory: `advanced/i2a/`*

- Learns an environment model and rolls imagined transitions.
- Aggregates imagined outcomes into policy/value decision layers.
- Targeted first for discrete-control tasks.

## MVE / STEVE
*Directory: `advanced/mve_steve/`*

- MVE: multi-step model-based value expansion targets.
- STEVE: uncertainty-weighted mixture of multi-horizon targets across model ensemble predictions.
