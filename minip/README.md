# Automated Incident Response Using Reinforcement Learning

A reinforcement learning-based system for automated security incident response that learns to detect and respond to cyber attacks (brute-force login attacks and ransomware-like behavior) without explicit detection rules.

## 🎯 Project Overview
python main.py train --episodes 500 --n-step --dueling --compare

This project implements an RL agent that:
- Observes noisy behavioral metrics (login rates, file access patterns, CPU usage, **rate-of-change features**)
- Learns to select defensive actions (block IP, lock account, terminate process, isolate host)
- Responds to attacks early while minimizing false positives
- Operates under uncertainty without knowing the true attack state

## 📁 Project Structure

```
minip/
├── main.py                    # Main entry point with CLI
├── requirements.txt           # Python dependencies
├── data/                      # Datasets
│   ├── Monday-WorkingHours.pcap_ISCX.csv   # CICIDS 2017 (benign)
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv  # CICIDS 2017 (attacks)
│   └── file.csv                            # CERT Insider Threat
├── src/
│   ├── config.py              # Configuration parameters
│   ├── preprocess.py          # Data preprocessing & feature extraction
│   ├── attack_simulator.py    # Probabilistic attack simulation
│   ├── incident_env.py        # Custom OpenAI Gym environment
│   ├── agent.py               # DQN agent (with N-step, dueling)
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation, statistics & visualization
├── models/                    # Saved model checkpoints
├── logs/                      # Training logs
└── figures/                   # Generated visualizations
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
python main.py all --episodes 500
```

This will:
1. Preprocess the datasets
2. Train the RL agent for 500 episodes
3. Generate training visualizations
4. Run a demo showing the trained agent

### 3. Individual Commands

```bash
# Preprocess datasets
python main.py preprocess

# Train the agent (with all enhancements)
python main.py train --episodes 1000 --n-step --dueling --compare

# Evaluate trained model
python main.py evaluate --analyze-policy

# Generate visualizations
python main.py visualize

# Run interactive demo
python main.py demo --interactive
```

## 🧠 Technical Details

### Enhanced Observation Space (10D)

The environment provides an **enhanced 10-dimensional observation space** for better attack detection:

| Feature | Description | Range |
|---------|-------------|-------|
| `login_rate` | Login attempts per window | [0, 200] |
| `file_access_rate` | File accesses per window | [0, 500] |
| `cpu_usage` | CPU usage percentage | [0, 100] |
| `login_delta` | **Rate of change** in login attempts | [-100, 100] |
| `file_delta` | **Rate of change** in file access | [-200, 200] |
| `cpu_delta` | **Rate of change** in CPU usage | [-50, 50] |
| `login_ma` | **Moving average** of login rate | [0, 200] |
| `file_ma` | **Moving average** of file rate | [0, 500] |
| `sustained_indicator` | **Sustained anomaly** indicator | [0, 1] |
| `normalized_time` | Episode progress | [0, 1] |

> **Note**: Rate-of-change features help detect attack **escalation** patterns.

### Attack Simulation

Two attack types modeled as probabilistic FSMs:

**Brute-Force Attack:**
```
Normal → Probing → Active → Compromised
```

**Ransomware Attack:**
```
Normal → Execution → Encryption → Data Loss
```

### Statistical Modeling

| Approach | Application |
|----------|-------------|
| **Poisson distributions** | Event counts (login attempts, file accesses) |
| **Local rate modeling** | Captures burstiness in network activity |
| **Normal distributions** | CPU usage with N(30,5) normal, N(80,5) attack |

> **Dataset Note**: CICIDS 2017 provides network flow features. `Total Fwd Packets` is used as a proxy for login attempts since explicit authentication logs are unavailable.

### DQN Agent Enhancements

| Feature | Description | Flag |
|---------|-------------|------|
| **Double DQN** | Reduces overestimation bias | Default |
| **N-step Returns** | Better temporal credit assignment | `--n-step` |
| **Dueling Architecture** | Separate value/advantage streams | `--dueling` |
| **Prioritized Replay** | Sample important experiences more | `--prioritized-replay` |

### Reward Structure

```python
rewards = {
    "early_containment": +50,    # Stopped attack in stage 1-2
    "late_containment": +20,     # Stopped attack in stage 3+
    "correct_no_action": +1,     # No action when no attack
    "false_positive": -10,       # Action when no attack
    "missed_attack": -30,        # Attack reached final state
    "step_penalty": -0.1         # Encourages efficiency
}
```

## 📊 Statistical Significance Testing

The project includes rigorous statistical analysis:

```bash
python main.py train --episodes 500 --compare
```

Outputs include:
- **Welch's t-test** with p-values
- **Cohen's d** effect size
- **95% confidence intervals**
- **Mann-Whitney U test** (non-parametric)

Example output:
```
DQN vs random:
  T-statistic: 8.4521
  P-value: 0.000001
  Cohen's d: 1.23
  95% CI: (12.45, 25.67)
  Significant: ✓
```

## 📈 Hyperparameter Sensitivity Analysis

Run sensitivity studies on key hyperparameters:

```python
from src.evaluate import HyperparameterAnalyzer
from src.train import Trainer

analyzer = HyperparameterAnalyzer()
results = analyzer.run_sensitivity_study(
    Trainer,
    param_name='learning_rate',
    param_values=[1e-4, 5e-4, 1e-3, 5e-3],
    num_episodes=200,
    num_seeds=3
)
analyzer.plot_sensitivity('learning_rate')
```

## 🔧 Training Options

```bash
python main.py train \
    --episodes 1000 \
    --attack-type random \
    --n-step \
    --n-steps 3 \
    --dueling \
    --checkpoint-dir models \
    --compare
```

| Option | Description | Default |
|--------|-------------|---------|
| `--episodes` | Training episodes | 500 |
| `--attack-type` | bruteforce, ransomware, both, random | random |
| `--n-step` | Enable N-step returns | False |
| `--n-steps` | N for N-step returns | 3 |
| `--dueling` | Use dueling architecture | False |
| `--no-enhanced` | Use 4D observation instead of 10D | False |
| `--compare` | Compare with baselines + statistics | False |

## 📚 References

- CICIDS 2017 Dataset: Canadian Institute for Cybersecurity
- CERT Insider Threat Dataset: Software Engineering Institute
- DQN: Mnih et al., "Human-level control through deep reinforcement learning"
- Double DQN: van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning"
- Dueling DQN: Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning"

## 🎓 Academic Rigor

This implementation includes features expected in academic work:

- ✅ **Poisson-based simulation** with empirical justification
- ✅ **Statistical significance testing** (t-tests, effect sizes)
- ✅ **Ablation study support** (baseline comparisons)
- ✅ **Hyperparameter sensitivity analysis**
- ✅ **Documented limitations** (proxy features, synthetic attacks)

## 📄 License

This project is for educational purposes as part of a mini project.
