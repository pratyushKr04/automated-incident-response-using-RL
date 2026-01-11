"""
Attack Simulator Module for Automated Incident Response.

This module implements probabilistic finite-state machines for:
- Brute-force login attacks: Normal → Probing → Active → Compromised
- Ransomware-like attacks: Normal → Execution → Encryption → Data Loss

Uses Poisson distributions for event-count modeling with local rate variations.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class BruteForceState(Enum):
    """States for brute-force attack progression."""
    NORMAL = 0
    PROBING = 1
    ACTIVE = 2
    COMPROMISED = 3


class RansomwareState(Enum):
    """States for ransomware attack progression."""
    NORMAL = 0
    EXECUTION = 1
    ENCRYPTION = 2
    DATA_LOSS = 3


@dataclass
class AttackObservation:
    """Observable metrics from attack simulation."""
    login_attempts: float      # Login attempts in time window
    file_access_rate: float    # File accesses in time window
    cpu_usage: float           # CPU usage percentage
    flow_duration: float       # Average flow duration
    is_attack_active: bool     # Ground truth (hidden from agent)
    attack_stage: int          # Current attack stage (hidden from agent)


class BruteForceAttack:
    """
    Simulates brute-force login attack behavior.
    
    Uses a probabilistic FSM where:
    - State transitions follow configured probabilities
    - Observable metrics are generated using Poisson distributions
    - Defender actions can influence state transitions
    """
    
    def __init__(self, config=None):
        """
        Initialize brute-force attack simulator.
        
        Args:
            config: AttackConfig object with parameters
        """
        if config is None:
            from config import get_config
            config = get_config().attack
        
        self.config = config
        self.state = BruteForceState.NORMAL
        self.steps_in_state = 0
        
        # Poisson rates for each state
        self.rates = {
            BruteForceState.NORMAL: config.bruteforce_rates["normal"],
            BruteForceState.PROBING: config.bruteforce_rates["probing"],
            BruteForceState.ACTIVE: config.bruteforce_rates["active"],
            BruteForceState.COMPROMISED: config.bruteforce_rates["compromised"]
        }
        
        # Parse transition probabilities
        self.transitions = {}
        for state_name, trans in config.bruteforce_transitions.items():
            state = BruteForceState[state_name.upper()]
            self.transitions[state] = {
                BruteForceState[k.upper()]: v for k, v in trans.items()
            }
    
    def reset(self) -> None:
        """Reset attack to initial state."""
        self.state = BruteForceState.NORMAL
        self.steps_in_state = 0
    
    def step(self, defender_action: Optional[int] = None) -> Tuple[float, bool]:
        """
        Advance attack by one time step.
        
        Args:
            defender_action: Action taken by defender (affects transitions)
            
        Returns:
            Tuple of (login_attempts, is_attack_contained)
        """
        # Generate observable: login attempts using Poisson
        base_rate = self.rates[self.state]
        
        # Add local rate variation (burstiness)
        rate_multiplier = np.random.lognormal(0, 0.3)
        effective_rate = base_rate * rate_multiplier
        
        login_attempts = np.random.poisson(effective_rate)
        
        # State transition
        contained = False
        
        if defender_action is not None:
            # Defender actions affect transition probabilities
            contained = self._apply_defender_action(defender_action)
        
        if not contained:
            # Natural state transition
            self._transition()
        
        self.steps_in_state += 1
        
        return float(login_attempts), contained
    
    def _transition(self) -> None:
        """Perform probabilistic state transition."""
        trans_probs = self.transitions[self.state]
        states = list(trans_probs.keys())
        probs = list(trans_probs.values())
        
        new_state = np.random.choice(
            [s.value for s in states],
            p=probs
        )
        new_state = BruteForceState(new_state)
        
        if new_state != self.state:
            self.state = new_state
            self.steps_in_state = 0
    
    def _apply_defender_action(self, action: int) -> bool:
        """
        Apply defender action and determine if attack is contained.
        
        Args:
            action: Defender action index
            
        Returns:
            True if attack is contained
        """
        # Action mapping:
        # 0: do_nothing
        # 1: block_ip
        # 2: lock_account
        # 3: terminate_process
        # 4: isolate_host
        
        if action == 0:  # Do nothing
            return False
        
        if self.state == BruteForceState.NORMAL:
            # No attack to contain
            return False
        
        # Effectiveness depends on attack stage and action
        containment_probs = {
            BruteForceState.PROBING: {1: 0.9, 2: 0.7, 3: 0.3, 4: 0.95},
            BruteForceState.ACTIVE: {1: 0.7, 2: 0.8, 3: 0.5, 4: 0.9},
            BruteForceState.COMPROMISED: {1: 0.3, 2: 0.4, 3: 0.6, 4: 0.8}
        }
        
        if self.state in containment_probs and action in containment_probs[self.state]:
            prob = containment_probs[self.state][action]
            if np.random.random() < prob:
                self.state = BruteForceState.NORMAL
                self.steps_in_state = 0
                return True
        
        return False
    
    @property
    def is_attacking(self) -> bool:
        """Check if attack is in progress."""
        return self.state != BruteForceState.NORMAL
    
    @property
    def is_compromised(self) -> bool:
        """Check if system is compromised."""
        return self.state == BruteForceState.COMPROMISED


class RansomwareAttack:
    """
    Simulates ransomware-like attack behavior.
    
    Uses file access patterns from CERT dataset modeling.
    """
    
    def __init__(self, config=None):
        """
        Initialize ransomware attack simulator.
        
        Args:
            config: AttackConfig object with parameters
        """
        if config is None:
            from config import get_config
            config = get_config().attack
        
        self.config = config
        self.state = RansomwareState.NORMAL
        self.steps_in_state = 0
        
        # Poisson rates for file access
        self.rates = {
            RansomwareState.NORMAL: config.ransomware_rates["normal"],
            RansomwareState.EXECUTION: config.ransomware_rates["execution"],
            RansomwareState.ENCRYPTION: config.ransomware_rates["encryption"],
            RansomwareState.DATA_LOSS: config.ransomware_rates["data_loss"]
        }
        
        # Parse transition probabilities
        self.transitions = {}
        for state_name, trans in config.ransomware_transitions.items():
            state = RansomwareState[state_name.upper()]
            self.transitions[state] = {
                RansomwareState[k.upper()]: v for k, v in trans.items()
            }
        
        # CPU usage parameters
        self.cpu_normal = (config.cpu_normal_mean, config.cpu_normal_std)
        self.cpu_attack = (config.cpu_attack_mean, config.cpu_attack_std)
    
    def reset(self) -> None:
        """Reset attack to initial state."""
        self.state = RansomwareState.NORMAL
        self.steps_in_state = 0
    
    def step(self, defender_action: Optional[int] = None) -> Tuple[float, float, bool]:
        """
        Advance attack by one time step.
        
        Args:
            defender_action: Action taken by defender
            
        Returns:
            Tuple of (file_access_rate, cpu_usage, is_attack_contained)
        """
        # Generate file access rate using Poisson
        base_rate = self.rates[self.state]
        rate_multiplier = np.random.lognormal(0, 0.2)
        effective_rate = base_rate * rate_multiplier
        
        file_access_rate = np.random.poisson(effective_rate)
        
        # Generate CPU usage
        if self.state == RansomwareState.NORMAL:
            cpu_usage = np.random.normal(*self.cpu_normal)
        else:
            # Blend between normal and attack CPU based on state
            attack_intensity = self.state.value / 3.0  # 0 to 1
            cpu_mean = (1 - attack_intensity) * self.cpu_normal[0] + \
                      attack_intensity * self.cpu_attack[0]
            cpu_usage = np.random.normal(cpu_mean, self.cpu_attack[1])
        
        cpu_usage = np.clip(cpu_usage, 0, 100)
        
        # Apply defender action
        contained = False
        if defender_action is not None:
            contained = self._apply_defender_action(defender_action)
        
        if not contained:
            self._transition()
        
        self.steps_in_state += 1
        
        return float(file_access_rate), float(cpu_usage), contained
    
    def _transition(self) -> None:
        """Perform probabilistic state transition."""
        trans_probs = self.transitions[self.state]
        states = list(trans_probs.keys())
        probs = list(trans_probs.values())
        
        new_state = np.random.choice(
            [s.value for s in states],
            p=probs
        )
        new_state = RansomwareState(new_state)
        
        if new_state != self.state:
            self.state = new_state
            self.steps_in_state = 0
    
    def _apply_defender_action(self, action: int) -> bool:
        """Apply defender action and determine containment."""
        if action == 0:
            return False
        
        if self.state == RansomwareState.NORMAL:
            return False
        
        # Effectiveness for ransomware
        containment_probs = {
            RansomwareState.EXECUTION: {1: 0.4, 2: 0.5, 3: 0.85, 4: 0.95},
            RansomwareState.ENCRYPTION: {1: 0.2, 2: 0.3, 3: 0.7, 4: 0.9},
            RansomwareState.DATA_LOSS: {1: 0.0, 2: 0.0, 3: 0.1, 4: 0.5}
        }
        
        if self.state in containment_probs and action in containment_probs[self.state]:
            prob = containment_probs[self.state][action]
            if np.random.random() < prob:
                self.state = RansomwareState.NORMAL
                self.steps_in_state = 0
                return True
        
        return False
    
    @property
    def is_attacking(self) -> bool:
        """Check if attack is in progress."""
        return self.state != RansomwareState.NORMAL
    
    @property
    def is_data_lost(self) -> bool:
        """Check if data loss has occurred."""
        return self.state == RansomwareState.DATA_LOSS


class CombinedAttackSimulator:
    """
    Combined attack simulator that can run brute-force and/or ransomware attacks.
    """
    
    def __init__(self, config=None, attack_type: str = "random"):
        """
        Initialize combined simulator.
        
        Args:
            config: Configuration object
            attack_type: "bruteforce", "ransomware", "both", or "random"
        """
        if config is None:
            from config import get_config
            config = get_config()
        
        self.config = config
        self.attack_type = attack_type
        
        self.bruteforce = BruteForceAttack(config.attack)
        self.ransomware = RansomwareAttack(config.attack)
        
        self.active_attack = None
        self._select_attack()
    
    def _select_attack(self) -> None:
        """Select which attack type to simulate."""
        if self.attack_type == "random":
            self.active_attack = np.random.choice(["bruteforce", "ransomware", "both", "none"])
        else:
            self.active_attack = self.attack_type
    
    def reset(self) -> None:
        """Reset all attacks."""
        self.bruteforce.reset()
        self.ransomware.reset()
        self._select_attack()
    
    def step(self, defender_action: Optional[int] = None) -> AttackObservation:
        """
        Advance simulation by one time step.
        
        Args:
            defender_action: Action taken by defender
            
        Returns:
            AttackObservation with all observable metrics
        """
        # Get metrics from both attack types
        if self.active_attack in ["bruteforce", "both"]:
            login_attempts, bf_contained = self.bruteforce.step(defender_action)
        else:
            login_attempts = np.random.poisson(2.0)  # Normal activity
            bf_contained = False
        
        if self.active_attack in ["ransomware", "both"]:
            file_access_rate, cpu_usage, rw_contained = self.ransomware.step(defender_action)
        else:
            file_access_rate = np.random.poisson(5.0)  # Normal activity
            cpu_usage = np.random.normal(30, 5)
            rw_contained = False
        
        # Flow duration (correlated with attack activity)
        if self.bruteforce.is_attacking:
            flow_duration = np.random.exponential(100)  # Shorter flows during attack
        else:
            flow_duration = np.random.exponential(500)  # Longer normal flows
        
        # Determine overall attack status
        is_attack_active = self.bruteforce.is_attacking or self.ransomware.is_attacking
        
        # Attack stage (max of both)
        attack_stage = max(self.bruteforce.state.value, self.ransomware.state.value)
        
        return AttackObservation(
            login_attempts=login_attempts,
            file_access_rate=file_access_rate,
            cpu_usage=np.clip(cpu_usage, 0, 100),
            flow_duration=flow_duration,
            is_attack_active=is_attack_active,
            attack_stage=attack_stage
        )
    
    @property
    def is_compromised(self) -> bool:
        """Check if system is in a terminal bad state."""
        return self.bruteforce.is_compromised or self.ransomware.is_data_lost
    
    @property
    def is_attacking(self) -> bool:
        """Check if any attack is active."""
        return self.bruteforce.is_attacking or self.ransomware.is_attacking


if __name__ == "__main__":
    # Test the attack simulators
    print("Testing Brute-Force Attack Simulator")
    print("=" * 50)
    
    bf = BruteForceAttack()
    for i in range(20):
        login_attempts, _ = bf.step()
        print(f"Step {i+1}: State={bf.state.name}, LoginAttempts={login_attempts:.0f}")
    
    print("\n\nTesting Ransomware Attack Simulator")
    print("=" * 50)
    
    rw = RansomwareAttack()
    for i in range(20):
        file_rate, cpu, _ = rw.step()
        print(f"Step {i+1}: State={rw.state.name}, FileRate={file_rate:.0f}, CPU={cpu:.1f}%")
    
    print("\n\nTesting Combined Simulator")
    print("=" * 50)
    
    sim = CombinedAttackSimulator(attack_type="both")
    for i in range(10):
        obs = sim.step()
        print(f"Step {i+1}: Login={obs.login_attempts:.0f}, Files={obs.file_access_rate:.0f}, "
              f"CPU={obs.cpu_usage:.1f}%, Attack={obs.is_attack_active}")
