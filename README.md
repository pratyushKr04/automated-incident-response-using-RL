# Automated Incident Response Using Deep Reinforcement Learning

An RL-based system that learns to detect and respond to cyber attacks—brute-force login attempts and ransomware—without hardcoded detection rules. Built as a mini project, inspired by *Finding Effective Security Strategies through Reinforcement Learning and Self-Play* (Hammar & Stadler, IEEE CNSM 2021).

## Project Overview

The agent observes noisy system metrics (login rates, file access patterns, CPU usage) and learns to pick defensive actions (block IP, lock account, terminate process, isolate host) that contain attacks early while keeping false positives low. It never sees the true attack state—only the same kind of noisy signals a real monitoring system would get.

**Key results (1000 episodes):**
- 75% success rate
- Beats all baselines (Snort, NIST 800-61, MITRE ATT&CK, threshold, random)
- Statistically significant improvements (p < 0.05, Cohen's d up to 5.55)

## Project Structure

```
minip/
├── main.py                    # CLI entry point (preprocess + train)
├── requirements.txt           # Python dependencies
├── report.tex                 # LaTeX project report
├── data/                      # Datasets
│   ├── Monday-WorkingHours.pcap_ISCX.csv   # CICIDS 2017 (benign)
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv  # CICIDS 2017 (attacks)
│   └── file.csv                            # CERT Insider Threat
├── src/
│   ├── config.py              # Configuration (loaded from extracted_params.json)
│   ├── preprocess.py          # Feature extraction from datasets
│   ├── attack_simulator.py    # Brute force & ransomware FSMs
│   ├── incident_env.py        # Custom Gymnasium environment (9D obs)
│   ├── agent.py               # Dueling Double DQN agent
│   ├── train.py               # Training loop + baseline comparison
│   └── evaluate.py            # Visualization & statistical tests
├── models/                    # Saved model weights
├── logs/                      # Training metrics (JSON)
└── figures/                   # Generated plots
```

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Preprocess datasets

```bash
python main.py preprocess
```

Extracts Poisson distribution parameters from CICIDS 2017 and CERT datasets, saves them to `extracted_params.json`.

### Train the agent

```bash
# Basic training (1000 episodes recommended)
python main.py train --episodes 1000

# Train and compare against all baselines
python main.py train --episodes 1000 --compare-baselines
```

Training automatically:
1. Runs preprocessing (if not done already)
2. Trains the Dueling Double DQN agent
3. Evaluates over 100 episodes
4. Generates all figures to `figures/`
5. Optionally compares against 6 baselines with statistical tests

### CLI options

| Option | Description | Default |
|--------|-------------|---------|
| `--episodes` | Number of training episodes | 500 |
| `--attack-type` | `bruteforce`, `ransomware`, `both`, `random` | `random` |
| `--compare-baselines` | Run baseline comparison with stats | off |
| `--n-step` | Enable N-step returns | off |
| `--n-steps` | Steps for N-step returns | 3 |
| `--prioritized-replay` | Prioritized experience replay | off |
| `--checkpoint-dir` | Model save directory | `models` |
| `--log-dir` | Log directory | `logs` |
| `--output-dir` | Figure output directory | `figures` |

## Technical Details

### Observation Space (9D)

| Feature | What it captures | Range |
|---------|-----------------|-------|
| `login_rate` | Raw login attempts per window | [0, 200] |
| `file_access_rate` | Raw file accesses per window | [0, 500] |
| `cpu_usage` | CPU utilization | [0, 100] |
| `login_delta` | Rate of change in logins | [-100, 100] |
| `file_delta` | Rate of change in file access | [-200, 200] |
| `cpu_delta` | Rate of change in CPU | [-50, 50] |
| `login_ma` | Moving avg of login rate (10-step) | [0, 200] |
| `file_ma` | Moving avg of file rate (10-step) | [0, 500] |
| `sustained_indicator` | How long metrics have been elevated | [0, 1] |

The delta and moving average features help the agent distinguish sudden attack-onset spikes from normal fluctuations.

### Action Space

| Action | Description | False positive cost |
|--------|-------------|-------------------|
| Do nothing | Continue monitoring | None |
| Block IP | Block suspicious source | Low |
| Lock account | Freeze compromised account | Medium |
| Terminate process | Kill suspicious process | Medium |
| Isolate host | Quarantine entire machine | High |

### Attack Simulation

Attacks are modeled as probabilistic finite state machines with parameters fitted from real datasets:

**Brute force (SSH credential stuffing):**
```
Idle → Probing → Active → Compromised
```

**Ransomware (file encryption):**
```
Idle → Execution → Encryption → Data Loss
```

Defensive actions can interrupt transitions with stage-dependent probability (earlier = more effective).

### Agent Architecture

- **Dueling Double DQN** — separates state value from action advantage
- **Network**: 128 → 64 → 32 (shared), then value stream (32 → 1) and advantage stream (32 → 5)
- **Experience replay**: 10,000 buffer, minibatch size 64
- **Target network**: synced every 10 episodes
- **Epsilon**: 1.0 → 0.01, decay factor 0.995

### Reward Structure

| Event | Reward |
|-------|--------|
| Early containment (stage 0–1) | +50 |
| Late containment (stage 2+) | +20 |
| Correct inaction | +1 |
| False positive | −10 |
| Missed attack (compromise) | −30 |
| Redundant action | −5 |
| Per-step cost | −0.1 |

### Baselines

Six baselines for comparison, each evaluated over 100 episodes with identical seeds:

1. **Random** — uniform random action selection
2. **Do-nothing** — always monitors, never acts
3. **Threshold** — fixed metric cutoffs (similar to basic SIEM rules)
4. **Snort-inspired** — signature-style thresholding rules
5. **NIST 800-61** — weighted impact scoring per NIST guidelines
6. **MITRE ATT&CK** — technique detection (T1110, T1486) with severity-based response

### Statistical Validation

All DQN-vs-baseline comparisons include:
- Independent samples t-test (p-values)
- Cohen's d effect size
- 95% confidence intervals

Results are saved to `logs/statistical_results.json`.

## Base Paper

> Hammar, K., & Stadler, R. (2021). *Finding Effective Security Strategies through Reinforcement Learning and Self-Play.* IEEE International Conference on Network and Service Management (CNSM), pp. 113–121.

Our work extends their approach by:
- Using a **custom Gymnasium environment** instead of CyberBattleSim
- Grounding attack simulations in **real dataset parameters** (CICIDS 2017, CERT)
- Focusing on **incident response actions** rather than abstract network defense
- Comparing against **industry-standard security frameworks** (Snort, NIST, MITRE)

## References

- Hammar & Stadler (2021) — *Finding Effective Security Strategies through RL and Self-Play* (base paper)
- CICIDS 2017 — Canadian Institute for Cybersecurity intrusion detection dataset
- CERT Insider Threat Dataset — Software Engineering Institute
- Mnih et al. (2015) — *Human-level control through deep reinforcement learning*
- Van Hasselt et al. (2016) — *Deep Reinforcement Learning with Double Q-learning*
- Wang et al. (2016) — *Dueling Network Architectures for Deep Reinforcement Learning*

## Team

| Name | Roll No |
|------|---------|
| Pratyush Kumar | 23BCS099 |
| Ritik Kumar Shahi | 23BCS110 |
| Saisha Bore | 23BCS116 |
| Aryan Talikoti | 23BCS018 |

## License

This project is for educational purposes as part of a mini project.
