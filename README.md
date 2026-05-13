# 🏥 BVSOP-MADRL — Blockchain Validator Selection & Optimization via Multi-Agent Deep Reinforcement Learning

---

## 📑 Table of Contents

1. [What Problem Are We Solving?](#-what-problem-are-we-solving)
2. [How We Solve It (The Big Picture)](#-how-we-solve-it-the-big-picture)
3. [Repository Structure](#-repository-structure)
4. [System Requirements](#-system-requirements)
5. [Environment Setup](#-environment-setup)
   - [Windows](#windows)
   - [macOS (Apple Silicon & Intel)](#macos-apple-silicon--intel)
   - [Linux (Ubuntu/Debian)](#linux-ubuntudebian)
6. [Running in Jupyter Notebook](#-running-in-jupyter-notebook)
7. [Running in VS Code](#-running-in-vs-code)
8. [Configuration Reference](#-configuration-reference)
9. [Code Architecture](#-code-architecture)
10. [Training the Agents](#-training-the-agents)
11. [Testing & Evaluation](#-testing--evaluation)
12. [Result Analysis](#-result-analysis)
13. [Saved Models](#-saved-models)
14. [Troubleshooting](#-troubleshooting)
15. [Research Background (For Juniors)](#-research-background-for-juniors)
16. [Glossary](#-glossary)

---

## 🔍 What Problem Are We Solving?

### The Real-World Scenario

Imagine multiple hospitals, pharmacies, insurance companies, and the Ministry of Public Health all need to **share medical data securely and quickly** over a shared blockchain network. Each participant (called an **Intelligent Participant, IP**) has:

- Medical transactions waiting in a queue (patient records, test results, drug prescriptions)
- Each transaction has a **urgency level** (how fast it needs to be processed), a **security level** (how many validators are needed), and a **queuing time** (how long it has been waiting)
- A shared pool of **blockchain validators** (nodes with different computational powers) to verify data blocks

### The Challenge: Three Conflicting Objectives

Every time a participant wants to send a data block, they must decide **three things simultaneously**:

| Decision | Effect |
|---|---|
| **How many transactions** to pack into one block | More → higher latency, lower cost per transaction |
| **How many validators** to involve | More → more secure, but slower and more expensive |
| **Whether to compress** the data | Compression → faster, but only appropriate for urgent (non-sensitive) data |

Minimizing latency, maximizing security, and minimizing cost are **fundamentally conflicting**. You cannot optimize all three perfectly at the same time.

### Why Not Just Use Maths?

The classical optimization approach requires solving a **non-convex Multi-Objective Optimization Problem (MOOP)** at every single time step. With 500 possible transaction counts × 40 possible validator counts × 3 compression choices = **60,000 combinations per agent per step**. With multiple agents and real-time requirements in healthcare, this is computationally infeasible in practice.

### Our Solution

We use **Multi-Agent Deep Reinforcement Learning (MARL)** — specifically the **MAD3QN algorithm** (Multi-Agent Dueling Double Deep Q-Network) — to train agents that **learn the optimal policy offline** and can make near-instant decisions at inference time.

---

## 🧠 How We Solve It (The Big Picture)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        IP-HealthChain System                        │
│                                                                     │
│  Hospital 1 (Agent 1)        Hospital 2 (Agent 2)                   │
│  ┌─────────────────┐         ┌─────────────────┐                    │
│  │  Observation:   │         │  Observation:   │                    │
│  │  - Queue state  │         │  - Queue state  │                    │
│  │  - Urgency lvls │         │  - Urgency lvls │                    │
│  │  - Security lvl │         │  - Security lvl │                    │
│  │  - Queuing time │         │  - Queuing time │                    │
│  │  - Validator x_i│         │  - Validator x_i│                    │
│  └────────┬────────┘         └────────┬────────┘                    │
│           │ action (d, tr, v)         │ action (d, tr, v)           │
│           └──────────────┬────────────┘                             │
│                          ▼                                          │
│              ┌───────────────────────┐                              │
│              │   Blockchain Network  │                              │
│              │  (Shared Validators)  │                              │
│              │  x₁, x₂, ..., x_M     │                              │
│              └──────────┬────────────┘                              │
│                         │                                           │
│              ┌──────────▼────────────┐                              │
│              │   Bipartite Reward    │                              │
│              │  r = δ·r_g + (1-δ)·r_l│                              │
│              │  r_g: global (shared) │                              │
│              │  r_l: local (per IP)  │                              │
│              └───────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```

### The Reward Signal (How the Agent Learns)

The reward has **two parts**:

**Global Reward (r_g)** — Shared equally by ALL agents. Encourages efficient use of the shared validator pool. If agents waste resources or send when none are available, they are penalized.

**Local Reward (r_l)** — Unique to each agent. Reflects how well the agent optimized its own latency, security, and cost for the transactions it just sent.

```
r = δ × r_g  +  (1 - δ) × r_l
```

Where `δ` (delta) controls the balance between cooperation (high δ) and competition (low δ).

---

## 📁 Repository Structure

```
BVSOP-MADRL/
│
├── BVSOP-MADRL-1.ipynb          ← Main Jupyter Notebook (all code lives here)
│
├── hospital_1_mad3qn_M25.pth    ← Trained model: Agent 1, M=25 validators
├── hospital_1_mad3qn_M40.pth    ← Trained model: Agent 1, M=40 validators
├── hospital_2_mad3qn_M25.pth    ← Trained model: Agent 2, M=25 validators
├── hospital_2_mad3qn_M40.pth    ← Trained model: Agent 2, M=40 validators
│
├── README.md                    ← This file
└── requirements.txt             ← All Python dependencies with pinned versions
```

> **Note on `.pth` files:** PyTorch model weights. These are the result of completed training runs and allow you to skip training and go straight to evaluation.

---

## 💻 System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU | None (CPU works) | CUDA-capable NVIDIA GPU |
| Disk Space | 2 GB | 5 GB |
| OS | Windows 10/11, macOS 12+, Ubuntu 20.04+ | Any of the above |

> **GPU Note:** The code automatically detects and uses a CUDA GPU if available via `torch.device("cuda" if torch.cuda.is_available() else "cpu")`. Training on CPU is fully supported but roughly 3–5× slower.

---

## ⚙️ Environment Setup

### Windows

**Step 1 — Install Python 3.11**

Download the installer from [python.org](https://www.python.org/downloads/). During installation:
- ✅ Check **"Add Python to PATH"**
- ✅ Check **"Install for all users"**

Verify in PowerShell:
```powershell
python --version
# Expected: Python 3.11.x
```

**Step 2 — Clone or download the repository**
```powershell
# If you have git installed:
git clone https://github.com/your-username/BVSOP-MADRL.git
cd BVSOP-MADRL

# OR simply download the ZIP from GitHub and extract it, then open PowerShell inside the folder.
```

**Step 3 — Create a virtual environment**
```powershell
python -m venv venv
venv\Scripts\activate
# Your prompt should now show: (venv) PS C:\...\BVSOP-MADRL>
```

**Step 4 — Install dependencies**
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

**Step 5 — (Optional) Install Jupyter**
```powershell
pip install jupyter notebook ipykernel
python -m ipykernel install --user --name=bvsop-env --display-name "BVSOP MADRL"
```

---

### macOS (Apple Silicon & Intel)

**Step 1 — Install Homebrew and Python**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.11
```

Verify:
```bash
python3.11 --version
# Expected: Python 3.11.x
```

**Step 2 — Clone the repository**
```bash
git clone https://github.com/your-username/BVSOP-MADRL.git
cd BVSOP-MADRL
```

**Step 3 — Create a virtual environment**
```bash
python3.11 -m venv venv
source venv/bin/activate
# Your prompt should now show: (venv) user@machine BVSOP-MADRL %
```

**Step 4 — Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3) Note:** PyTorch supports Apple's Metal Performance Shaders (MPS) backend. The code uses CUDA detection, but MPS will not auto-activate. If you want GPU acceleration on Apple Silicon, change the device line in the notebook:
> ```python
> device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
> ```

**Step 5 — (Optional) Install Jupyter**
```bash
pip install jupyter notebook ipykernel
python -m ipykernel install --user --name=bvsop-env --display-name "BVSOP MADRL"
```

---

### Linux (Ubuntu/Debian)

**Step 1 — Install Python and build tools**
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3.11 python3.11-venv python3.11-dev build-essential git -y
```

Verify:
```bash
python3.11 --version
```

**Step 2 — Clone the repository**
```bash
git clone https://github.com/your-username/BVSOP-MADRL.git
cd BVSOP-MADRL
```

**Step 3 — Create a virtual environment**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

**Step 4 — Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Step 5 — (Optional) CUDA Setup for NVIDIA GPU**

First verify your CUDA version:
```bash
nvidia-smi
# Note the "CUDA Version" shown in the top-right corner
```

If CUDA 12.x is available, PyTorch from `requirements.txt` should work. If you need a specific CUDA build, visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) and replace the torch line accordingly.

**Step 6 — (Optional) Install Jupyter**
```bash
pip install jupyter notebook ipykernel
python -m ipykernel install --user --name=bvsop-env --display-name "BVSOP MADRL"
```

---

## 📓 Running in Jupyter Notebook

**Step 1 — Activate your virtual environment** (if not already active)

```bash
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

**Step 2 — Launch Jupyter**
```bash
jupyter notebook
```
This opens a browser tab at `http://localhost:8888`.

**Step 3 — Open the notebook**

Click on `BVSOP-MADRL-1.ipynb` in the browser file tree.

**Step 4 — Select the correct kernel**

In the notebook menu: `Kernel → Change kernel → BVSOP MADRL`

If you skipped the ipykernel installation step, select any available Python 3 kernel that uses the venv.

**Step 5 — Run cells in order**

The notebook is divided into the following sections. Run them **top to bottom**:

| Section | What it Does |
|---|---|
| **Cell 1: Configuration & Imports** | Sets all hyperparameters, imports libraries, defines the `XI_VECTOR` of validator compute powers |
| **Cell 2: Environment (`IPHealthChainEnv`)** | Defines the simulation world — queues, validators, state transitions, reward computation |
| **Cell 3: Neural Network (`BranchingDuelingDQN`)** | Defines the Dueling DQN architecture with separate heads for each action sub-space |
| **Cell 4: Agents** | Defines MAD3QN, RS (Random), SB (Static), and ES (Exhaustive Search) agents |
| **Cell 5: Training Loop** | Runs the main training, saves `.pth` model files, records history |
| **Cell 6: Evaluation** | Loads a saved model and runs inference on a fixed environment |
| **Cell 7: Plotting** | Generates all result visualizations |

> **Tip:** Use `Kernel → Restart & Run All` for a clean full run. Expected training time: ~15–30 minutes on CPU for 3500 episodes.

---

## 🖥️ Running in VS Code

**Step 1 — Install VS Code**

Download from [code.visualstudio.com](https://code.visualstudio.com/).

**Step 2 — Install required extensions**

Open VS Code, press `Ctrl+Shift+X` (or `Cmd+Shift+X` on Mac), and install:
- **Python** (by Microsoft)
- **Jupyter** (by Microsoft)

**Step 3 — Open the project folder**

```
File → Open Folder → select the BVSOP-MADRL folder
```

**Step 4 — Select the Python interpreter**

Press `Ctrl+Shift+P` → type `Python: Select Interpreter` → choose the one that shows `venv` in its path, e.g.:
- Windows: `.\venv\Scripts\python.exe`
- macOS/Linux: `./venv/bin/python`

**Step 5 — Open and run the notebook**

Click on `BVSOP-MADRL-1.ipynb` in the Explorer panel. VS Code will render it as a notebook. Click **"Select Kernel"** in the top right → choose `venv (Python 3.11)`.

Run cells individually with the ▶ button, or run all with `Run All` in the toolbar.

> **Tip:** VS Code shows a GPU indicator in the bottom status bar if CUDA is detected. Look for `Python 3.11.x 64-bit ('venv')` confirmation that you are using the right environment.

---

## ⚙️ Configuration Reference

All parameters are defined in the `CONFIG` dictionary at the top of the notebook. Here is a complete explanation:

### Blockchain & Physics Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `w` | `0.5e6` Hz | Channel bandwidth |
| `SNR_d_db` | `10` dB | Downlink signal-to-noise ratio |
| `SNR_u_db` | `12` dB | Uplink signal-to-noise ratio |
| `r_d` | `1.2e6` bps | Downlink transmission rate (BM → Validators) |
| `r_u` | `1.3e6` bps | Uplink transmission rate (Validators → BM) |
| `xi_hat` | `0.5e6` bits | Verification feedback size (ξ̂) |
| `zeta` | `500` bytes | Average transaction size (ζ) |
| `q` | `4.0` | Network scale indicator for security function |
| `kappa` | `1.0` | Security coefficient (κ) |
| `psi` | `0.001` | Consensus latency coefficient (ψ) |
| `G_required` | `100.0` | Computation required for block verification (K) |

### System Scale Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `T_r` / `X_max` | `500` | Maximum transactions per block |
| `M` | `40` | Total number of available validators |
| `N_agents` | `2` | Number of intelligent participants (hospitals) |

> **Switching Settings:** To reproduce the paper's "Setting 1" (M=25), change `"M": 40` → `"M": 25` and uncomment the `XI_VECTOR` for M=25. The code will automatically use the correct pre-trained model file (`_M25.pth`).

### Reward Weighting Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `alpha` | `0.33` | Weight for latency objective |
| `beta` | `0.33` | Weight for security objective |
| `gamma` | `0.34` | Weight for cost objective |
| `delta` | `0.2` | Global vs local reward balance (0=fully competitive, 1=fully cooperative) |

### Application-Level Thresholds

| Parameter | Default | Meaning |
|---|---|---|
| `u_th` | `0.5` | Urgency threshold for compression eligibility |
| `a_th` | `5` | Maximum allowed queuing time (zombie threshold) |
| `l_p` | `0.5` | Latency penalty coefficient |
| `s_p` | `0.5` | Security penalty coefficient |
| `d_p` | `0.5` | Wrong compression decision penalty |
| `U_p` | `0.5` | Insufficient resource utilization penalty |
| `epsilon_error` | `0.2` | Accepted security error margin (ε) |

### RL Hyperparameters

| Parameter | Default | Meaning |
|---|---|---|
| `LR` | `3e-4` | Learning rate for RMSprop optimizer |
| `GAMMA` | `0.99` | Discount factor (λ) — how much to value future rewards |
| `BATCH_SIZE` | `128` | Number of experiences sampled per learning step |
| `MEMORY_SIZE` | `50000` | Replay buffer capacity |
| `EPSILON_START` | `1.0` | Initial exploration rate (100% random actions) |
| `EPSILON_END` | `0.01` | Final exploration rate (1% random actions) |
| `EPSILON_DECAY` | `59500` | Number of steps to linearly decay epsilon |
| `TAU` | `0.001` | Soft update rate for target network |
| `EPISODES` | `3500` | Total training episodes |
| `STEPS_PER_EP` | `20` | Environment steps per episode |
| `HIDDEN_SIZE` | `512` | Neurons per hidden layer in the neural network |

---

## 🏗️ Code Architecture

### `IPHealthChainEnv` — The Simulation Environment

This class simulates the blockchain network world. Think of it as a game environment.

```
IPHealthChainEnv
│
├── __init__()         — Sets up validators, computes normalization bounds
├── _init_resources()  — Assigns random or fixed computational power (x_i) to validators
├── reset()            — Starts a fresh episode with a new random queue state
├── _get_obs()         — Builds the observation vector each agent sees
├── compute_security() — S(v) = κ · v^q
├── compute_latency()  — L(tr, v, ζ) = download + verify + consensus + upload
├── compute_cost()     — C(tr, v) = Σx_i / tr
└── step(actions)      — Takes actions from all agents, computes rewards, advances state
```

**What is the observation vector?**

Each agent sees a flattened vector containing:
1. Its **transaction queue** — up to `X_max` transactions, each with `(urgency, security, queuing_time)` — shape `(X_max × 3,)`
2. The **shared validator resources** — each validator's compute power normalized — shape `(M × 2,)`

Total observation size ≈ `500 × 3 + 40 × 2 = 1580` floats per agent.

**What is the zombie prevention algorithm?**

If a transaction has been waiting longer than `a_th` steps, it becomes a "zombie" — it risks never being processed because higher-urgency transactions keep jumping the queue. The environment **forces** a send action when zombies are detected, overriding the agent's idle decision. This prevents starvation of low-urgency medical data.

---

### `BranchingDuelingDQN` — The Neural Network

```
Input: observation vector (1580-dim)
    ↓
[Conv1D 1→16, k=3]  [ReLU]
    ↓
[Conv1D 16→32, k=3] [ReLU]
    ↓
[Flatten → conv_out_size]
    ↓ ─────────────────────────────────────────────────────
    │                    │                    │            │
[Value Stream V]  [Advantage A_d]  [Advantage A_tr]  [Advantage A_v]
[Linear→512→ReLU] [Linear→512→ReLU][Linear→512→ReLU] [Linear→512→ReLU]
[Linear→1]        [Linear→3]       [Linear→500]      [Linear→40]
    │                    │                    │            │
    └──────── Q_d = V + (A_d - mean(A_d)) ───┘
              Q_tr = V + (A_tr - mean(A_tr))
              Q_v = V + (A_v - mean(A_v))
```

**Why three output heads?** The action space is multi-discrete: `d ∈ {0,1,2}`, `tr ∈ {1..500}`, `v ∈ {1..40}`. A single head over the joint action space would require `3×500×40 = 60,000` outputs, which is intractable. The branching architecture handles each sub-action independently.

**Why Dueling?** The dueling architecture separates the **value of a state** (V) from the **advantage of a specific action** (A). This makes learning more stable because the network can update the state-value estimate even when it doesn't take an action, allowing faster convergence.

**Why Double DQN?** Standard DQN overestimates Q-values due to using the same network for both action selection and evaluation. Double DQN uses the **online network** to *select* the best action and the **target network** to *evaluate* it, eliminating this bias.

---

### The Four Competing Agents

| Agent | Strategy | Purpose |
|---|---|---|
| `MAD3QNAgent` | Learns optimal policy via experience replay + neural network | **Our proposed method** |
| `RSAgent` | Picks actions uniformly at random | Lower-bound baseline; represents no intelligence |
| `SBAgent` | Always picks the same fixed action | Baseline for zero-adaptation behavior |
| `ESAgent` | Exhaustively evaluates all possible actions at each step | Near-optimal greedy baseline (infeasible at scale) |

---

## 🚀 Training the Agents

### Quick Start

Open the notebook and run all cells in order. Training begins at the cell labeled **"Main Training Loop"**.

### What Happens During Training

1. The environment resets — fresh random transaction queues and validator resources
2. Each agent observes its local state (partial observability)
3. With probability ε (epsilon), the agent picks a **random action** (exploration); otherwise it picks the **greedy action** from its Q-network (exploitation)
4. The environment executes all actions simultaneously, computes bipartite rewards, and transitions to the next state
5. Each experience tuple `(obs, action, reward, next_obs, done)` is stored in the **replay buffer**
6. Every step, a random mini-batch of 128 experiences is sampled to update the Q-network (this breaks temporal correlations)
7. The **target network** is soft-updated toward the online network every step: `θ_target ← τ·θ_online + (1-τ)·θ_target`
8. Epsilon decays linearly over 59,500 steps until it reaches 0.01

### Progress Monitoring

Every 100 episodes, the console prints:
```
Episode 100/3500 | H1 Reward: 45.2341 | H2 Reward: 44.8901 | Resources: 67.3%
```

Healthy training shows **gradually increasing rewards** and **increasing resource utilization** over time. Expect convergence around episode 2000–3000.

### Saved Outputs

After training completes, two model files are saved:
```
hospital_1_mad3qn_M40.pth   ← Agent 1's learned weights
hospital_2_mad3qn_M40.pth   ← Agent 2's learned weights
```

---

## 🧪 Testing & Evaluation

### Loading a Pre-Trained Model

The evaluation cell creates a **fixed-resource environment** using the `XI_VECTOR` from the paper (so results are deterministic and reproducible) and loads saved weights:

```python
eval_env = IPHealthChainEnv(CONFIG, fixed_resources=XI_VECTOR)
eval_agents[0].online_net.load_state_dict(
    torch.load("hospital_1_mad3qn_M40.pth", map_location=device)
)
eval_agents[0].epsilon = 0.0   # ← Disables all exploration; purely greedy
```

### What the Output Means

```
----------------------------------------------------------------------------------
Selected: [0 0 1 0 0 1 0 0 1 0 0 0 0 1 0 0 0 0 1 0 ... ]
m: 8
n: 312
Utility U: 0.7842311203479767
----------------------------------------------------------------------------------
```

| Field | Meaning |
|---|---|
| `Selected` | Binary mask of length M — a `1` means that validator was chosen |
| `m` | Number of validators selected for this block |
| `n` | Number of transactions packed into this block |
| `Utility U` | The final bipartite reward value (0–1 range, higher is better) |

---

## 📊 Result Analysis

### Training Reward Curve

After training, plot the reward history to verify convergence:

```python
import matplotlib.pyplot as plt
import numpy as np

window = 50
for i in range(CONFIG["N_agents"]):
    rewards = history["agent_rewards"][i]
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(smoothed, label=f"Hospital {i+1}")

plt.xlabel("Training Episode")
plt.ylabel("Average Reward per Step")
plt.title(f"MAD3QN Training Convergence (M={CONFIG['M']})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"training_reward_M{CONFIG['M']}.png", dpi=150)
plt.show()
```

**What to look for:**
- Rewards should start low (~10–20) and climb to ~70–80
- Both agents should converge to similar reward levels (cooperative behavior)
- The curve should flatten and stabilize — that is convergence

### Resource Utilization Over Training

```python
resources = history["resources"]
smoothed_res = np.convolve(resources, np.ones(100)/100, mode='valid')
plt.plot(smoothed_res, color='green')
plt.axhline(y=100, color='red', linestyle='--', label='Max Resources')
plt.xlabel("Training Episode")
plt.ylabel("Resource Utilization (%)")
plt.title("Computational Resource Utilization During Training")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**What to look for:** Utilization should trend upward toward 100% as agents learn to cooperate in using the shared validator pool efficiently.

### Comparing All Four Policies

To reproduce the paper's benchmark comparison (Figure 5 in the paper), run all four agent types on the same fixed environment for 100 testing episodes and compare their accumulated rewards:

```python
policies = {
    "MAD3QN": eval_agents,
    "RS": [RSAgent(CONFIG, eval_env) for _ in range(CONFIG["N_agents"])],
    "SB": [SBAgent(CONFIG, eval_env) for _ in range(CONFIG["N_agents"])],
}

results = {}
TEST_EPISODES = 100

for policy_name, policy_agents in policies.items():
    all_rewards = []
    for ep in range(TEST_EPISODES):
        obs = eval_env.reset()
        ep_reward = 0
        for _ in range(CONFIG["STEPS_PER_EP"]):
            actions = [a.select_action(obs[i]) for i, a in enumerate(policy_agents)]
            obs, rewards, _, _ = eval_env.step(actions)
            ep_reward += np.mean(rewards)
        all_rewards.append(ep_reward / CONFIG["STEPS_PER_EP"])
    results[policy_name] = all_rewards
    print(f"{policy_name}: Avg Reward = {np.mean(all_rewards):.4f}")
```

### Interpreting Results Against the Paper

| Metric | Paper Result | What It Means |
|---|---|---|
| MAD3QN avg reward | ~75 (small scale) | Best overall policy |
| ES avg reward | ~41 | Good greedy baseline, but too slow for real use |
| RS avg reward | ~21 | Random is surprisingly better than static |
| SB avg reward | ~10.5 | Worst — blindly ignores all system state |
| Latency reduction vs ES | up to 46.6% | In emergency (high-urgency) scenarios |
| Average latency reduction | ~29.24% | Across all tested configurations |

---

## 💾 Saved Models

Four pre-trained model files are included:

| File | Setting | Agents | Episodes Trained |
|---|---|---|---|
| `hospital_1_mad3qn_M25.pth` | M=25 validators, Setting 1 | Agent 1 | 3500 |
| `hospital_2_mad3qn_M25.pth` | M=25 validators, Setting 1 | Agent 2 | 3500 |
| `hospital_1_mad3qn_M40.pth` | M=40 validators, Setting 2 | Agent 1 | 3500 |
| `hospital_2_mad3qn_M40.pth` | M=40 validators, Setting 2 | Agent 2 | 3500 |

To load a specific model, update the path in the evaluation cell:

```python
# For M=25 Setting 1:
eval_agents[0].online_net.load_state_dict(
    torch.load("hospital_1_mad3qn_M25.pth", map_location=device)
)

# For M=40 Setting 2 (default):
eval_agents[0].online_net.load_state_dict(
    torch.load("hospital_1_mad3qn_M40.pth", map_location=device)
)
```

**Remember:** When switching between M=25 and M=40, also update `CONFIG["M"]` and the `XI_VECTOR` accordingly before running any cell.

---

## 🔧 Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`
Your virtual environment is not activated. Run `source venv/bin/activate` (macOS/Linux) or `venv\Scripts\activate` (Windows) and retry.

### `FileNotFoundError: hospital_1_mad3qn_M40.pth`
You are trying to load a model that hasn't been trained yet. Either run the training cell first, or ensure the `.pth` files from the repository are in the same directory as the notebook.

### CUDA errors like `RuntimeError: CUDA out of memory`
Your GPU doesn't have enough VRAM. Add this line to force CPU usage:
```python
device = torch.device("cpu")
```

### Notebook kernel crashes immediately
This usually means insufficient RAM. Try reducing `MEMORY_SIZE` from `50000` to `20000` and `HIDDEN_SIZE` from `512` to `128`.

### Training reward stuck at ~10 and not improving
Epsilon may be decaying too fast, or learning rate is too high. Try:
```python
CONFIG["EPSILON_DECAY"] = 100000  # Slower decay
CONFIG["LR"] = 1e-4               # Lower learning rate
```

### `ValueError: operands could not be broadcast together`
This happens if `CONFIG["M"]` and the length of `XI_VECTOR` don't match. Make sure `XI_VECTOR` is sliced to `[:CONFIG["M"]]` — the code already does this, but verify no manual edits broke it.

### Jupyter not finding the venv kernel
Re-register the kernel:
```bash
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install ipykernel
python -m ipykernel install --user --name=bvsop-env --display-name "BVSOP MADRL"
jupyter notebook
```

---

## 📚 Research Background (For Juniors)

If you are new to this topic, here is a quick mental model for each technology used:

**Blockchain** — A distributed ledger where data is grouped into blocks and verified by multiple nodes (validators) before being added. No single party controls it. Think of it as a shared Google Doc that no one can secretly edit — every change requires approval from multiple trusted witnesses.

**Reinforcement Learning (RL)** — A training paradigm where an agent learns by trial and error. It takes actions in an environment, receives a reward signal (good/bad), and gradually learns which actions lead to higher long-term rewards. Like training a dog with treats, but the dog is a neural network and the tricks are blockchain configuration decisions.

**Multi-Agent RL (MARL)** — Multiple agents learning simultaneously in the same environment. They can cooperate (sharing a global reward) or compete (pursuing individual rewards). The challenge: from each agent's perspective, the environment appears non-stationary because other agents are also changing their behaviour.

**Deep Q-Network (DQN)** — Uses a neural network to approximate the Q-function: "given this state, what is the expected future reward of taking this action?" Instead of storing a table (which would be astronomically large), the network generalizes across similar states.

**Dueling DQN** — Splits the Q-network into two streams: one estimates how good the current state is overall (V), and one estimates how much better/worse each specific action is compared to average (A). This is more efficient because many states don't require distinguishing between actions.

**Double DQN** — Fixes the overestimation bias in standard DQN by using two separate networks: one chooses the action, the other evaluates it. This leads to more conservative, accurate Q-value estimates.

**Dec-POMDP** — Decentralized Partially Observable Markov Decision Process. Each agent only sees part of the global state (partial observability). Agents make decisions independently (decentralized) without communicating directly. The "Markov" part means the future depends only on the current state, not the full history.

---

## 📖 Glossary

| Term | Definition |
|---|---|
| **IP** | Intelligent Participant — a hospital, pharmacy, etc. acting as a blockchain node |
| **Validator** | A computational node that verifies blockchain transactions |
| **Block** | A batch of transactions packaged together for blockchain submission |
| **MOOP** | Multi-Objective Optimization Problem |
| **MARL** | Multi-Agent Reinforcement Learning |
| **MAD3QN** | Multi-Agent Dueling Double Deep Q-Network — the proposed algorithm |
| **D3QN** | Dueling Double Deep Q-Network (single-agent version) |
| **Dec-POMDP** | Decentralized Partially Observable Markov Decision Process |
| **Bipartite Reward** | The paper's novel two-part reward (global + local) that enables implicit cooperation |
| **Zombie Transaction** | A transaction that has been waiting longer than the threshold `a_th` and risks never being processed |
| **Exhaustive Search (ES)** | A brute-force policy that evaluates all possible actions and picks the best — optimal but slow |
| **Random Selection (RS)** | A policy that picks actions uniformly at random |
| **Static-Based (SB)** | A policy that always picks the same fixed action regardless of state |
| **DPoS** | Delegated Proof of Stake — the consensus mechanism used; validators are selected by computational resources |
| **CSLR** | Cost-Security-Latency-Resource — the combined optimization objective of the system |
| **ε-greedy** | Exploration strategy: with probability ε take a random action, otherwise take the best known action |
| **Replay Buffer** | A memory bank storing past experiences `(state, action, reward, next_state)` for random sampling during training |
| **Soft Update** | Slowly blending online network weights into the target network: `θ_target ← τ·θ_online + (1-τ)·θ_target` |

---

*Last updated: May 2026*