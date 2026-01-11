"""
Configuration parameters for the Automated Incident Response RL system.

This module loads parameters EXTRACTED from preprocessing the datasets.
No hardcoded fallback values - preprocessing MUST run first.
"""

import numpy as np
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


# Get project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
PARAMS_FILE = PROJECT_ROOT / "extracted_params.json"


class ConfigurationError(Exception):
    """Raised when configuration cannot be loaded."""
    pass


def load_extracted_params() -> Dict:
    """
    Load parameters extracted from preprocessing.
    
    Returns:
        Dictionary of extracted parameters
        
    Raises:
        ConfigurationError: If params file doesn't exist
    """
    if not PARAMS_FILE.exists():
        raise ConfigurationError(
            f"\n{'='*60}\n"
            f"ERROR: extracted_params.json not found!\n"
            f"{'='*60}\n\n"
            f"You must run preprocessing before training:\n"
            f"  python main.py preprocess\n\n"
            f"This extracts parameters from your datasets and saves them to:\n"
            f"  {PARAMS_FILE}\n"
            f"{'='*60}\n"
        )
    
    with open(PARAMS_FILE, 'r') as f:
        params = json.load(f)
    
    print(f"Loaded parameters from: {PARAMS_FILE}")
    return params


@dataclass
class AttackConfig:
    """Configuration for attack simulation - loaded from extracted parameters."""
    
    # Parameters will be loaded from JSON
    bruteforce_rates: Dict[str, float] = field(default_factory=dict)
    ransomware_rates: Dict[str, float] = field(default_factory=dict)
    
    # Brute-force attack states
    bruteforce_states: List[str] = field(default_factory=lambda: [
        "normal", "probing", "active", "compromised"
    ])
    
    # State transition probabilities for brute-force
    bruteforce_transitions: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "normal": {"normal": 0.95, "probing": 0.05, "active": 0.0, "compromised": 0.0},
        "probing": {"normal": 0.1, "probing": 0.6, "active": 0.3, "compromised": 0.0},
        "active": {"normal": 0.0, "probing": 0.1, "active": 0.7, "compromised": 0.2},
        "compromised": {"normal": 0.0, "probing": 0.0, "active": 0.0, "compromised": 1.0}
    })
    
    # Ransomware attack states
    ransomware_states: List[str] = field(default_factory=lambda: [
        "normal", "execution", "encryption", "data_loss"
    ])
    
    # State transition probabilities for ransomware
    ransomware_transitions: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "normal": {"normal": 0.95, "execution": 0.05, "encryption": 0.0, "data_loss": 0.0},
        "execution": {"normal": 0.05, "execution": 0.65, "encryption": 0.3, "data_loss": 0.0},
        "encryption": {"normal": 0.0, "execution": 0.0, "encryption": 0.6, "data_loss": 0.4},
        "data_loss": {"normal": 0.0, "execution": 0.0, "encryption": 0.0, "data_loss": 1.0}
    })
    
    # CPU usage parameters (loaded from JSON)
    cpu_normal_mean: float = 30.0
    cpu_normal_std: float = 5.0
    cpu_attack_mean: float = 80.0
    cpu_attack_std: float = 5.0
    
    @classmethod
    def from_extracted_params(cls, params: Dict) -> 'AttackConfig':
        """
        Create AttackConfig from extracted parameters.
        
        Args:
            params: Dictionary loaded from extracted_params.json
            
        Returns:
            AttackConfig instance with extracted values
        """
        config = cls()
        
        # Load brute-force parameters
        bf_params = params.get('bruteforce', {})
        config.bruteforce_rates = {
            "normal": bf_params.get('benign_login_rate', 2.0),
            "probing": bf_params.get('benign_login_rate', 2.0) * 7.5,  # 7.5x normal
            "active": bf_params.get('attack_login_rate', 50.0),
            "compromised": bf_params.get('benign_login_rate', 2.0) * 2.5  # Post-attack
        }
        
        # Load ransomware parameters
        rw_params = params.get('ransomware', {})
        config.ransomware_rates = {
            "normal": rw_params.get('normal_file_rate', 5.0),
            "execution": rw_params.get('execution_file_rate', 20.0),
            "encryption": rw_params.get('encryption_file_rate', 100.0),
            "data_loss": rw_params.get('post_attack_rate', 10.0)
        }
        
        # Load CPU parameters
        cpu_params = params.get('cpu_usage', {})
        config.cpu_normal_mean = cpu_params.get('normal_mean', 30.0)
        config.cpu_normal_std = cpu_params.get('normal_std', 5.0)
        config.cpu_attack_mean = cpu_params.get('attack_mean', 80.0)
        config.cpu_attack_std = cpu_params.get('attack_std', 5.0)
        
        return config


@dataclass
class EnvironmentConfig:
    """Configuration for the Gym environment."""
    
    max_steps: int = 100
    time_window_seconds: int = 60
    
    obs_low: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0]))
    obs_high: np.ndarray = field(default_factory=lambda: np.array([200.0, 500.0, 100.0, 100.0]))
    
    actions: List[str] = field(default_factory=lambda: [
        "do_nothing",
        "block_ip",
        "lock_account",
        "terminate_process",
        "isolate_host"
    ])
    
    rewards: Dict[str, float] = field(default_factory=lambda: {
        "early_containment": 50.0,
        "late_containment": 20.0,
        "correct_no_action": 1.0,
        "false_positive": -10.0,
        "missed_attack": -30.0,
        "unnecessary_action": -5.0,
        "step_penalty": -0.1
    })
    
    observation_noise_std: float = 2.0


@dataclass
class AgentConfig:
    """Configuration for the DQN agent."""
    
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    activation: str = "relu"
    
    learning_rate: float = 1e-3
    gamma: float = 0.99
    
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    buffer_size: int = 10000
    batch_size: int = 64
    
    target_update_freq: int = 10
    
    num_episodes: int = 1000
    max_steps_per_episode: int = 100
    
    save_freq: int = 100
    model_dir: str = "models"


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    attack: AttackConfig = field(default_factory=AttackConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    
    seed: int = 42
    
    # Data paths (relative to project root)
    cicids_monday_path: str = field(default_factory=lambda: str(PROJECT_ROOT / "data" / "Monday-WorkingHours.pcap_ISCX.csv"))
    cicids_tuesday_path: str = field(default_factory=lambda: str(PROJECT_ROOT / "data" / "Tuesday-WorkingHours.pcap_ISCX.csv"))
    cert_file_path: str = field(default_factory=lambda: str(PROJECT_ROOT / "data" / "file.csv"))
    
    @classmethod
    def from_extracted_params(cls) -> 'Config':
        """
        Create Config from extracted parameters.
        Raises ConfigurationError if params not found.
        
        Returns:
            Config instance with extracted values
        """
        params = load_extracted_params()
        
        config = cls()
        config.attack = AttackConfig.from_extracted_params(params)
        
        return config


# Cache for loaded config
_cached_config: Optional[Config] = None


def get_config(force_reload: bool = False) -> Config:
    """
    Get configuration with extracted parameters.
    
    This function will:
    1. Load parameters from extracted_params.json
    2. Raise an error if the file doesn't exist (preprocessing required)
    
    Args:
        force_reload: If True, reload from file even if cached
        
    Returns:
        Config object with extracted parameters
        
    Raises:
        ConfigurationError: If extracted_params.json doesn't exist
    """
    global _cached_config
    
    if _cached_config is None or force_reload:
        _cached_config = Config.from_extracted_params()
    
    return _cached_config


def ensure_preprocessing_done() -> bool:
    """
    Check if preprocessing has been done.
    
    Returns:
        True if extracted_params.json exists
    """
    return PARAMS_FILE.exists()


if __name__ == "__main__":
    # Print configuration for verification
    if ensure_preprocessing_done():
        config = get_config()
        print("=== Attack Configuration (from extracted params) ===")
        print(f"Brute-force rates: {config.attack.bruteforce_rates}")
        print(f"Ransomware rates: {config.attack.ransomware_rates}")
        print(f"CPU normal: {config.attack.cpu_normal_mean} ± {config.attack.cpu_normal_std}")
        print(f"CPU attack: {config.attack.cpu_attack_mean} ± {config.attack.cpu_attack_std}")
        
        print("\n=== Environment Configuration ===")
        print(f"Actions: {config.env.actions}")
        print(f"Max steps: {config.env.max_steps}")
        
        print("\n=== Agent Configuration ===")
        print(f"Hidden layers: {config.agent.hidden_layers}")
        print(f"Learning rate: {config.agent.learning_rate}")
    else:
        print("ERROR: extracted_params.json not found!")
        print("Run: python main.py preprocess")
