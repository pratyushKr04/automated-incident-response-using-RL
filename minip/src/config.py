"""
Configuration parameters for the Automated Incident Response RL system.

This module contains all hyperparameters and configuration settings for:
- Attack simulation parameters
- Environment settings
- RL agent hyperparameters
- Training configuration
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class AttackConfig:
    """Configuration for attack simulation."""
    
    # Brute-force attack parameters (derived from CICIDS 2017)
    bruteforce_states: List[str] = field(default_factory=lambda: [
        "normal", "probing", "active", "compromised"
    ])
    
    # Poisson rates for login attempts per time window (λ values)
    bruteforce_rates: Dict[str, float] = field(default_factory=lambda: {
        "normal": 2.0,      # ~2 login attempts per window (benign)
        "probing": 15.0,    # ~15 attempts (scanning)
        "active": 50.0,     # ~50 attempts (active attack)
        "compromised": 5.0  # Reduced after success
    })
    
    # State transition probabilities for brute-force
    bruteforce_transitions: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "normal": {"normal": 0.95, "probing": 0.05, "active": 0.0, "compromised": 0.0},
        "probing": {"normal": 0.1, "probing": 0.6, "active": 0.3, "compromised": 0.0},
        "active": {"normal": 0.0, "probing": 0.1, "active": 0.7, "compromised": 0.2},
        "compromised": {"normal": 0.0, "probing": 0.0, "active": 0.0, "compromised": 1.0}
    })
    
    # Ransomware attack parameters (derived from CERT dataset)
    ransomware_states: List[str] = field(default_factory=lambda: [
        "normal", "execution", "encryption", "data_loss"
    ])
    
    # Poisson rates for file access per time window
    ransomware_rates: Dict[str, float] = field(default_factory=lambda: {
        "normal": 5.0,       # ~5 file accesses per window
        "execution": 20.0,   # Increased activity
        "encryption": 100.0, # Rapid file access
        "data_loss": 10.0    # Post-encryption
    })
    
    # State transition probabilities for ransomware
    ransomware_transitions: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "normal": {"normal": 0.95, "execution": 0.05, "encryption": 0.0, "data_loss": 0.0},
        "execution": {"normal": 0.05, "execution": 0.65, "encryption": 0.3, "data_loss": 0.0},
        "encryption": {"normal": 0.0, "execution": 0.0, "encryption": 0.6, "data_loss": 0.4},
        "data_loss": {"normal": 0.0, "execution": 0.0, "encryption": 0.0, "data_loss": 1.0}
    })
    
    # CPU usage parameters (modeled as latent variable)
    cpu_normal_mean: float = 30.0
    cpu_normal_std: float = 5.0
    cpu_attack_mean: float = 80.0
    cpu_attack_std: float = 5.0


@dataclass
class EnvironmentConfig:
    """Configuration for the Gym environment."""
    
    # Episode settings
    max_steps: int = 100
    time_window_seconds: int = 60  # Each step represents 1 minute
    
    # Observation space bounds
    obs_low: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0]))
    obs_high: np.ndarray = field(default_factory=lambda: np.array([200.0, 500.0, 100.0, 100.0]))
    # [login_rate, file_access_rate, cpu_usage, time_step_normalized]
    
    # Action space
    actions: List[str] = field(default_factory=lambda: [
        "do_nothing",
        "block_ip",
        "lock_account",
        "terminate_process",
        "isolate_host"
    ])
    
    # Reward structure
    rewards: Dict[str, float] = field(default_factory=lambda: {
        "early_containment": 50.0,      # Stopped attack early
        "late_containment": 20.0,       # Stopped attack but damage done
        "correct_no_action": 1.0,       # Correctly did nothing during normal
        "false_positive": -10.0,        # Took action during normal state
        "missed_attack": -30.0,         # Attack progressed to final state
        "unnecessary_action": -5.0,     # Redundant defensive action
        "step_penalty": -0.1            # Small penalty per step to encourage efficiency
    })
    
    # Noise parameters for observations
    observation_noise_std: float = 2.0


@dataclass
class AgentConfig:
    """Configuration for the DQN agent."""
    
    # Network architecture
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    activation: str = "relu"
    
    # Learning parameters
    learning_rate: float = 1e-3
    gamma: float = 0.99  # Discount factor
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Experience replay
    buffer_size: int = 10000
    batch_size: int = 64
    
    # Target network
    target_update_freq: int = 10  # Episodes
    
    # Training
    num_episodes: int = 1000
    max_steps_per_episode: int = 100
    
    # Model saving
    save_freq: int = 100  # Save every N episodes
    model_dir: str = "models"


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    attack: AttackConfig = field(default_factory=AttackConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Data paths
    cicids_monday_path: str = "data/Monday-WorkingHours.pcap_ISCX.csv"
    cicids_tuesday_path: str = "data/Tuesday-WorkingHours.pcap_ISCX.csv"
    cert_file_path: str = "data/file.csv"


# Default configuration instance
default_config = Config()


def get_config() -> Config:
    """Get the default configuration."""
    return Config()


if __name__ == "__main__":
    # Print configuration for verification
    config = get_config()
    print("=== Attack Configuration ===")
    print(f"Brute-force states: {config.attack.bruteforce_states}")
    print(f"Brute-force rates: {config.attack.bruteforce_rates}")
    print(f"Ransomware states: {config.attack.ransomware_states}")
    print(f"Ransomware rates: {config.attack.ransomware_rates}")
    
    print("\n=== Environment Configuration ===")
    print(f"Actions: {config.env.actions}")
    print(f"Max steps: {config.env.max_steps}")
    print(f"Rewards: {config.env.rewards}")
    
    print("\n=== Agent Configuration ===")
    print(f"Hidden layers: {config.agent.hidden_layers}")
    print(f"Learning rate: {config.agent.learning_rate}")
    print(f"Epsilon decay: {config.agent.epsilon_start} -> {config.agent.epsilon_end}")
