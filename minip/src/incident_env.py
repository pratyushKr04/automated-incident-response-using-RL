"""
Custom OpenAI Gym Environment for Automated Incident Response.

This environment simulates a security incident where an RL agent must:
- Observe noisy behavioral metrics (login rates, file access, CPU usage)
- Select defensive actions (block IP, lock account, terminate process, isolate host)
- Learn to respond quickly while minimizing false positives

The attack type and stage are hidden from the agent - it must infer
the threat level from observable metrics alone.

TECHNICAL NOTES:
- Feature extraction uses network flow features (Total Fwd Packets, Flow Duration)
  from CICIDS 2017 as proxies for login attempts, since explicit authentication
  logs are not available. This is a reasonable approximation documented in
  cybersecurity literature.
- Rate-of-change features are included to help the agent detect attack escalation.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
from collections import deque

from attack_simulator import CombinedAttackSimulator, AttackObservation


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""
    total_reward: float = 0.0
    steps: int = 0
    attacks_detected: int = 0
    attacks_contained: int = 0
    false_positives: int = 0
    missed_attacks: int = 0
    data_loss_events: int = 0


class IncidentResponseEnv(gym.Env):
    """
    Custom Gym environment for automated incident response.
    
    Enhanced Observation Space (10 dimensions):
        - login_rate: Number of login attempts in time window [0, 200]
        - file_access_rate: Number of file accesses in time window [0, 500]
        - cpu_usage: CPU usage percentage [0, 100]
        - login_rate_delta: Rate of change in login attempts [-100, 100]
        - file_rate_delta: Rate of change in file access [-200, 200]
        - cpu_delta: Rate of change in CPU usage [-50, 50]
        - login_moving_avg: Moving average of login rate (smoothed) [0, 200]
        - file_moving_avg: Moving average of file rate (smoothed) [0, 500]
        - sustained_high_activity: Normalized indicator of sustained anomaly [0, 1]
    
    Action Space:
        0: do_nothing - Take no action
        1: block_ip - Block the source IP address
        2: lock_account - Lock the targeted user account
        3: terminate_process - Kill suspicious processes
        4: isolate_host - Isolate the affected host from network
    
    Rewards:
        - Early containment: +50
        - Late containment: +20
        - Correct no-action: +1
        - False positive: -10
        - Data loss: -30
        - Step penalty: -0.1 (encourages efficiency)
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        config=None,
        attack_type: str = "random",
        render_mode: Optional[str] = None,
        use_enhanced_features: bool = True,
        moving_avg_window: int = 5
    ):
        """
        Initialize the environment.
        
        Args:
            config: Configuration object
            attack_type: Type of attack to simulate
            render_mode: Rendering mode ("human" or "ansi")
            use_enhanced_features: Whether to use enhanced observation space
            moving_avg_window: Window size for moving average calculations
        """
        super().__init__()
        
        if config is None:
            from config import get_config
            config = get_config()
        
        self.config = config
        self.attack_type = attack_type
        self.render_mode = render_mode
        self.use_enhanced_features = use_enhanced_features
        self.moving_avg_window = moving_avg_window
        
        # Initialize attack simulator
        self.simulator = CombinedAttackSimulator(config, attack_type)
        
        # Define observation space based on feature mode
        if use_enhanced_features:
            # Enhanced 9-dimensional observation space (no time feature for real-world applicability)
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, 0.0, -100.0, -200.0, -50.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([200.0, 500.0, 100.0, 100.0, 200.0, 50.0, 200.0, 500.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
            self.obs_dim = 9
        else:
            # Original 4-dimensional observation space
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([200.0, 500.0, 100.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
            self.obs_dim = 4
        
        # Define action space
        self.action_space = spaces.Discrete(5)
        self.action_names = [
            "do_nothing",
            "block_ip", 
            "lock_account",
            "terminate_process",
            "isolate_host"
        ]
        
        # Environment state
        self.current_step = 0
        self.max_steps = config.env.max_steps
        self.last_observation: Optional[AttackObservation] = None
        self.episode_stats = EpisodeStats()
        
        # Track actions taken (for detecting redundant actions)
        self.actions_taken = set()
        
        # Noise for observations
        self.obs_noise_std = config.env.observation_noise_std
        
        # Reward parameters
        self.rewards = config.env.rewards
        
        # History for rendering and feature calculation
        self.history: List[Dict] = []
        
        # History buffers for rate-of-change and moving average calculations
        self.login_history = deque(maxlen=moving_avg_window)
        self.file_history = deque(maxlen=moving_avg_window)
        self.cpu_history = deque(maxlen=moving_avg_window)
        
        # Previous values for delta calculation
        self.prev_login = 0.0
        self.prev_file = 0.0
        self.prev_cpu = 30.0
        
        # Sustained high activity counter
        self.high_activity_counter = 0
        self.high_activity_threshold = 3  # Steps of sustained high activity
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options
            
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        
        # Reset simulator
        self.simulator.reset()
        
        # Reset state
        self.current_step = 0
        self.actions_taken = set()
        self.episode_stats = EpisodeStats()
        self.history = []
        
        # Reset history buffers
        self.login_history.clear()
        self.file_history.clear()
        self.cpu_history.clear()
        
        # Reset previous values
        self.prev_login = 0.0
        self.prev_file = 0.0
        self.prev_cpu = 30.0
        
        # Reset sustained activity counter
        self.high_activity_counter = 0
        
        # Get initial observation
        self.last_observation = self.simulator.step(None)
        
        # Initialize history with initial values
        self.login_history.append(self.last_observation.login_attempts)
        self.file_history.append(self.last_observation.file_access_rate)
        self.cpu_history.append(self.last_observation.cpu_usage)
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step.
        
        Args:
            action: Action to take (0-4)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        
        # Store previous values for delta calculation
        self.prev_login = self.last_observation.login_attempts if self.last_observation else 0
        self.prev_file = self.last_observation.file_access_rate if self.last_observation else 0
        self.prev_cpu = self.last_observation.cpu_usage if self.last_observation else 30
        
        # Record previous state for reward calculation
        was_attacking = self.simulator.is_attacking
        was_compromised = self.simulator.is_compromised
        prev_stage = self.last_observation.attack_stage if self.last_observation else 0
        
        # Execute action and advance simulation
        self.last_observation = self.simulator.step(action)
        
        # Update history buffers
        self.login_history.append(self.last_observation.login_attempts)
        self.file_history.append(self.last_observation.file_access_rate)
        self.cpu_history.append(self.last_observation.cpu_usage)
        
        # Update sustained high activity counter
        self._update_high_activity_counter()
        
        # Calculate reward
        reward = self._calculate_reward(
            action=action,
            was_attacking=was_attacking,
            was_compromised=was_compromised,
            prev_stage=prev_stage
        )
        
        # Update statistics
        self.episode_stats.total_reward += reward
        self.episode_stats.steps = self.current_step
        
        # Get observation
        obs = self._get_observation()
        
        # Check termination conditions
        terminated = self.simulator.is_compromised  # Bad ending
        truncated = self.current_step >= self.max_steps
        
        # Store history for rendering
        self.history.append({
            "step": self.current_step,
            "action": self.action_names[action],
            "reward": reward,
            "login_rate": self.last_observation.login_attempts,
            "file_rate": self.last_observation.file_access_rate,
            "cpu_usage": self.last_observation.cpu_usage,
            "is_attack": self.last_observation.is_attack_active
        })
        
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, terminated, truncated, info
    
    def _update_high_activity_counter(self) -> None:
        """Update the sustained high activity counter."""
        # Define thresholds for "high activity"
        login_threshold = 25.0
        file_threshold = 40.0
        
        current_login = self.last_observation.login_attempts
        current_file = self.last_observation.file_access_rate
        
        if current_login > login_threshold or current_file > file_threshold:
            self.high_activity_counter = min(self.high_activity_counter + 1, 10)
        else:
            self.high_activity_counter = max(self.high_activity_counter - 1, 0)
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation with noise.
        
        Returns:
            Noisy observation array
        """
        if self.last_observation is None:
            return np.zeros(self.obs_dim, dtype=np.float32)
        
        current_login = self.last_observation.login_attempts
        current_file = self.last_observation.file_access_rate
        current_cpu = self.last_observation.cpu_usage
        
        if self.use_enhanced_features:
            # Calculate rate of change (delta)
            login_delta = current_login - self.prev_login
            file_delta = current_file - self.prev_file
            cpu_delta = current_cpu - self.prev_cpu
            
            # Calculate moving averages
            login_ma = np.mean(list(self.login_history)) if self.login_history else current_login
            file_ma = np.mean(list(self.file_history)) if self.file_history else current_file
            
            # Sustained high activity indicator (normalized)
            sustained_indicator = self.high_activity_counter / self.high_activity_threshold
            sustained_indicator = min(sustained_indicator, 1.0)
            
            obs = np.array([
                current_login,           # Raw login rate
                current_file,            # Raw file access rate
                current_cpu,             # Raw CPU usage
                login_delta,             # Rate of change in login
                file_delta,              # Rate of change in file access
                cpu_delta,               # Rate of change in CPU
                login_ma,                # Moving average of login
                file_ma,                 # Moving average of file access
                sustained_indicator,     # Sustained anomaly indicator
            ], dtype=np.float32)
            
            # Add observation noise (except to sustained indicator)
            noise = np.random.normal(0, self.obs_noise_std, 8)
            obs[:8] += noise
        else:
            # Original 4D observation
            obs = np.array([
                current_login,
                current_file,
                current_cpu,
                self.current_step / self.max_steps
            ], dtype=np.float32)
            
            # Add observation noise (except to time)
            noise = np.random.normal(0, self.obs_noise_std, 3)
            obs[:3] += noise
        
        # Clip to valid ranges
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        
        return obs
    
    def _calculate_reward(
        self,
        action: int,
        was_attacking: bool,
        was_compromised: bool,
        prev_stage: int
    ) -> float:
        """
        Calculate reward based on action and state transitions.
        
        Args:
            action: Action taken
            was_attacking: Whether attack was active before action
            was_compromised: Whether system was compromised before
            prev_stage: Previous attack stage
            
        Returns:
            Reward value
        """
        reward = self.rewards["step_penalty"]  # Small step cost
        
        is_attacking = self.simulator.is_attacking
        is_compromised = self.simulator.is_compromised
        current_stage = self.last_observation.attack_stage
        
        # Case 1: Action during normal operation
        if not was_attacking and action == 0:
            reward += self.rewards["correct_no_action"]
        elif not was_attacking and action > 0:
            # False positive - took action when no attack
            reward += self.rewards["false_positive"]
            self.episode_stats.false_positives += 1
        
        # Case 2: Attack was active
        elif was_attacking:
            self.episode_stats.attacks_detected += 1
            
            if action > 0:
                # Took defensive action
                if action in self.actions_taken:
                    # Redundant action
                    reward += self.rewards["unnecessary_action"]
                else:
                    self.actions_taken.add(action)
                    
                    # Check if attack was contained
                    if not is_attacking:
                        # Successfully contained!
                        self.episode_stats.attacks_contained += 1
                        
                        if prev_stage <= 1:
                            reward += self.rewards["early_containment"]
                        else:
                            reward += self.rewards["late_containment"]
                    else:
                        # Action taken but attack continues
                        reward += 2.0  # Small positive for trying
            else:
                # No action during attack
                if is_compromised and not was_compromised:
                    # Attack progressed to final state
                    reward += self.rewards["missed_attack"]
                    self.episode_stats.missed_attacks += 1
                    self.episode_stats.data_loss_events += 1
        
        # Case 3: System was compromised
        if is_compromised and not was_compromised:
            reward += self.rewards["missed_attack"]
        
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        return {
            "step": self.current_step,
            "is_attack_active": self.last_observation.is_attack_active if self.last_observation else False,
            "attack_stage": self.last_observation.attack_stage if self.last_observation else 0,
            "is_compromised": self.simulator.is_compromised,
            "episode_stats": {
                "total_reward": self.episode_stats.total_reward,
                "attacks_detected": self.episode_stats.attacks_detected,
                "attacks_contained": self.episode_stats.attacks_contained,
                "false_positives": self.episode_stats.false_positives,
                "missed_attacks": self.episode_stats.missed_attacks
            }
        }
    
    def render(self) -> Optional[str]:
        """Render the environment."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            self._render_human()
        return None
    
    def _render_human(self) -> None:
        """Print human-readable output."""
        if not self.last_observation:
            return
        
        obs = self.last_observation
        print(f"\n--- Step {self.current_step}/{self.max_steps} ---")
        print(f"Login attempts: {obs.login_attempts:.0f}")
        print(f"File access rate: {obs.file_access_rate:.0f}")
        print(f"CPU usage: {obs.cpu_usage:.1f}%")
        print(f"Attack active: {'YES' if obs.is_attack_active else 'NO'}")
        print(f"Sustained high activity: {self.high_activity_counter}")
        print(f"Total reward: {self.episode_stats.total_reward:.2f}")
    
    def _render_ansi(self) -> str:
        """Return ANSI string representation."""
        if not self.last_observation:
            return ""
        
        obs = self.last_observation
        attack_indicator = "🔴" if obs.is_attack_active else "🟢"
        
        return (
            f"Step {self.current_step:3d} | "
            f"Login: {obs.login_attempts:5.0f} | "
            f"Files: {obs.file_access_rate:5.0f} | "
            f"CPU: {obs.cpu_usage:5.1f}% | "
            f"{attack_indicator}"
        )
    
    def close(self) -> None:
        """Clean up resources."""
        pass


class MultiAttackEnv(IncidentResponseEnv):
    """
    Extended environment that can handle multiple concurrent attacks.
    """
    
    def __init__(self, num_hosts: int = 3, **kwargs):
        """
        Initialize multi-host environment.
        
        Args:
            num_hosts: Number of hosts to monitor
            **kwargs: Arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.num_hosts = num_hosts
        
        # Extended observation space for multiple hosts
        # Each host has: [login_rate, file_access_rate, cpu_usage]
        self.observation_space = spaces.Box(
            low=np.zeros(num_hosts * 3 + 1, dtype=np.float32),
            high=np.concatenate([
                np.tile([200.0, 500.0, 100.0], num_hosts),
                [1.0]  # normalized time
            ]).astype(np.float32),
            dtype=np.float32
        )
        
        # Extended action space: action for each host
        self.action_space = spaces.MultiDiscrete([5] * num_hosts)
        
        # Multiple simulators
        self.simulators = [
            CombinedAttackSimulator(self.config, self.attack_type)
            for _ in range(num_hosts)
        ]


# Register the environment
def register_envs():
    """Register custom environments with Gymnasium."""
    try:
        gym.register(
            id="IncidentResponse-v0",
            entry_point="incident_env:IncidentResponseEnv",
            max_episode_steps=100
        )
        gym.register(
            id="IncidentResponse-v1",
            entry_point="incident_env:MultiAttackEnv",
            max_episode_steps=100
        )
    except Exception:
        pass  # Already registered


if __name__ == "__main__":
    # Test the environment
    print("Testing Incident Response Environment")
    print("=" * 60)
    
    # Test enhanced features
    print("\n--- Testing Enhanced Features (10D observation) ---")
    env = IncidentResponseEnv(attack_type="random", render_mode="human", use_enhanced_features=True)
    
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Observation space: {env.observation_space}")
    print(f"Initial observation:\n  Login={obs[0]:.1f}, Files={obs[1]:.1f}, CPU={obs[2]:.1f}")
    print(f"  Delta Login={obs[3]:.1f}, Delta Files={obs[4]:.1f}, Delta CPU={obs[5]:.1f}")
    print(f"  MA Login={obs[6]:.1f}, MA Files={obs[7]:.1f}")
    print(f"  Sustained={obs[8]:.2f}, Time={obs[9]:.2f}")
    
    total_reward = 0
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step + 1}")
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Episode stats: {env.episode_stats}")
    
    env.close()
    
    # Test original features
    print("\n--- Testing Original Features (4D observation) ---")
    env = IncidentResponseEnv(attack_type="random", use_enhanced_features=False)
    obs, _ = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Observation: {obs}")
    env.close()
