"""
DQN Agent for Automated Incident Response using TensorFlow/Keras.

This module implements:
- Deep Q-Network (DQN) with experience replay
- Double DQN for reduced overestimation
- N-step returns for better temporal credit assignment
- Epsilon-greedy exploration with decay
- Target network for stable training
- Prioritized experience replay (optional)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from collections import deque, namedtuple
import random
from typing import List, Tuple, Optional, Dict
import os


# Experience tuple for replay buffer
Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done'])

# N-step experience for n-step returns
NStepExperience = namedtuple('NStepExperience',
    ['state', 'action', 'n_step_reward', 'next_state', 'done', 'n'])


def create_q_network(
    state_size: int,
    action_size: int,
    hidden_layers: List[int] = [128, 64, 32],
    use_dueling: bool = False
) -> keras.Model:
    """
    Create Q-Network using Keras functional API.
    
    Args:
        state_size: Dimension of state space
        action_size: Number of possible actions
        hidden_layers: Sizes of hidden layers
        use_dueling: Whether to use dueling architecture
        
    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=(state_size,))
    x = inputs
    
    # Shared layers
    for i, units in enumerate(hidden_layers[:-1]):
        x = layers.Dense(units, activation='relu', 
                        kernel_initializer='glorot_uniform',
                        name=f'shared_{i}')(x)
        x = layers.Dropout(0.1)(x)
    
    if use_dueling:
        # Dueling architecture: separate value and advantage streams
        last_hidden = hidden_layers[-1]
        
        # Value stream
        value = layers.Dense(last_hidden, activation='relu', name='value_hidden')(x)
        value = layers.Dense(1, name='value_output')(value)
        
        # Advantage stream
        advantage = layers.Dense(last_hidden, activation='relu', name='advantage_hidden')(x)
        advantage = layers.Dense(action_size, name='advantage_output')(advantage)
        
        # Combine: Q = V + (A - mean(A))
        outputs = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
    else:
        # Standard architecture
        x = layers.Dense(hidden_layers[-1], activation='relu', 
                        kernel_initializer='glorot_uniform',
                        name='hidden_last')(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(action_size, name='q_values')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='q_network')
    return model


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    
    Stores experiences and samples random minibatches for training.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add experience to buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)


class NStepReplayBuffer:
    """
    N-step replay buffer for better temporal credit assignment.
    """
    
    def __init__(self, capacity: int = 10000, n_steps: int = 3, gamma: float = 0.99):
        """
        Initialize n-step buffer.
        
        Args:
            capacity: Maximum buffer size
            n_steps: Number of steps for n-step returns
            gamma: Discount factor
        """
        self.buffer = deque(maxlen=capacity)
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_steps)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add experience and compute n-step returns when ready."""
        self.n_step_buffer.append(Experience(state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) == self.n_steps or done:
            n_step_reward = 0.0
            for i, exp in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * exp.reward
                if exp.done:
                    break
            
            first_exp = self.n_step_buffer[0]
            last_exp = self.n_step_buffer[-1]
            
            self.buffer.append(NStepExperience(
                state=first_exp.state,
                action=first_exp.action,
                n_step_reward=n_step_reward,
                next_state=last_exp.next_state,
                done=last_exp.done,
                n=len(self.n_step_buffer)
            ))
            
            if done:
                self.n_step_buffer.clear()
    
    def sample(self, batch_size: int) -> List[NStepExperience]:
        """Sample random batch of n-step experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def reset_episode(self) -> None:
        """Reset n-step buffer at episode end."""
        while len(self.n_step_buffer) > 0:
            n_step_reward = 0.0
            for i, exp in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * exp.reward
            
            first_exp = self.n_step_buffer[0]
            last_exp = self.n_step_buffer[-1]
            
            self.buffer.append(NStepExperience(
                state=first_exp.state,
                action=first_exp.action,
                n_step_reward=n_step_reward,
                next_state=last_exp.next_state,
                done=True,
                n=len(self.n_step_buffer)
            ))
            self.n_step_buffer.popleft()
    
    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized experience replay buffer.
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        super().__init__(capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: float = 1.0
    ) -> None:
        """Add experience with priority."""
        self.buffer.append(Experience(state, action, reward, next_state, done))
        self.priorities.append(priority ** self.alpha)
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], np.ndarray, List[int]]:
        """Sample batch with prioritized sampling."""
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), 
                                   p=probs, replace=False)
        
        experiences = [self.buffer[i] for i in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return experiences, weights, indices.tolist()
    
    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha


class DQNAgent:
    """
    Deep Q-Network Agent for incident response using TensorFlow/Keras.
    
    Features:
    - Experience replay for stable training
    - Target network for reducing overestimation
    - Double DQN for action selection
    - N-step returns for better credit assignment
    - Epsilon-greedy exploration with decay
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        config=None,
        use_prioritized_replay: bool = False,
        use_n_step: bool = False,
        n_steps: int = 3,
        use_dueling: bool = False
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            config: AgentConfig object
            use_prioritized_replay: Whether to use prioritized replay
            use_n_step: Whether to use n-step returns
            n_steps: Number of steps for n-step returns
            use_dueling: Whether to use dueling architecture
        """
        if config is None:
            from config import get_config
            config = get_config().agent
        
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        self.use_n_step = use_n_step
        self.n_steps = n_steps
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Using GPU: {gpus[0].name}")
            self.device = '/GPU:0'
        else:
            print("Using CPU")
            self.device = '/CPU:0'
        
        # Q-Networks
        self.q_network = create_q_network(
            state_size, action_size, 
            config.hidden_layers,
            use_dueling=use_dueling
        )
        
        self.target_network = create_q_network(
            state_size, action_size,
            config.hidden_layers,
            use_dueling=use_dueling
        )
        
        # Copy weights to target network
        self.target_network.set_weights(self.q_network.get_weights())
        
        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate=config.learning_rate)
        
        # Replay buffer
        if use_n_step:
            self.memory = NStepReplayBuffer(config.buffer_size, n_steps, config.gamma)
            print(f"Using {n_steps}-step returns")
        elif use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(config.buffer_size)
            print("Using prioritized experience replay")
        else:
            self.memory = ReplayBuffer(config.buffer_size)
        
        self.use_prioritized = use_prioritized_replay
        
        # Exploration parameters
        self.epsilon = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        
        # Training parameters
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.target_update_freq = config.target_update_freq
        
        # Training statistics
        self.training_step = 0
        self.episode_count = 0
        self.losses = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            training: Whether in training mode
            
        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        # Greedy action
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        q_values = self.q_network(state_tensor, training=False)
        return int(tf.argmax(q_values[0]).numpy())
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    @tf.function
    def _train_step_graph(self, states, actions, rewards, next_states, dones, weights, gamma_n):
        """TensorFlow graph-mode training step for speed."""
        with tf.GradientTape() as tape:
            # Current Q values
            current_q_all = self.q_network(states, training=True)
            actions_one_hot = tf.one_hot(actions, self.action_size)
            current_q = tf.reduce_sum(current_q_all * actions_one_hot, axis=1)
            
            # Target Q values (Double DQN)
            next_q_online = self.q_network(next_states, training=False)
            next_actions = tf.argmax(next_q_online, axis=1)
            next_actions_one_hot = tf.one_hot(next_actions, self.action_size)
            
            next_q_target = self.target_network(next_states, training=False)
            next_q = tf.reduce_sum(next_q_target * next_actions_one_hot, axis=1)
            
            target_q = rewards + gamma_n * next_q * (1.0 - dones)
            
            # Huber loss (smooth L1)
            td_error = current_q - target_q
            loss = tf.reduce_mean(weights * tf.where(
                tf.abs(td_error) < 1.0,
                0.5 * tf.square(td_error),
                tf.abs(td_error) - 0.5
            ))
        
        # Update network
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        return loss, tf.abs(td_error)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        if self.use_prioritized and not self.use_n_step:
            experiences, weights, indices = self.memory.sample(self.batch_size)
            weights = np.array(weights, dtype=np.float32)
        else:
            experiences = self.memory.sample(self.batch_size)
            weights = np.ones(len(experiences), dtype=np.float32)
            indices = None
        
        # Prepare batch data
        if self.use_n_step:
            states = np.array([e.state for e in experiences], dtype=np.float32)
            actions = np.array([e.action for e in experiences], dtype=np.int32)
            rewards = np.array([e.n_step_reward for e in experiences], dtype=np.float32)
            next_states = np.array([e.next_state for e in experiences], dtype=np.float32)
            dones = np.array([float(e.done) for e in experiences], dtype=np.float32)
            n_values = np.array([e.n for e in experiences], dtype=np.float32)
            gamma_n = self.gamma ** n_values
        else:
            states = np.array([e.state for e in experiences], dtype=np.float32)
            actions = np.array([e.action for e in experiences], dtype=np.int32)
            rewards = np.array([e.reward for e in experiences], dtype=np.float32)
            next_states = np.array([e.next_state for e in experiences], dtype=np.float32)
            dones = np.array([float(e.done) for e in experiences], dtype=np.float32)
            gamma_n = np.full(len(experiences), self.gamma, dtype=np.float32)
        
        # Convert to tensors
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        next_states = tf.convert_to_tensor(next_states)
        dones = tf.convert_to_tensor(dones)
        weights = tf.convert_to_tensor(weights)
        gamma_n = tf.convert_to_tensor(gamma_n)
        
        # Training step
        loss, td_errors = self._train_step_graph(
            states, actions, rewards, next_states, dones, weights, gamma_n
        )
        
        # Update priorities
        if self.use_prioritized and indices is not None:
            self.memory.update_priorities(indices, td_errors.numpy() + 1e-6)
        
        self.training_step += 1
        loss_value = float(loss.numpy())
        self.losses.append(loss_value)
        
        return loss_value
    
    def update_target_network(self) -> None:
        """Copy weights from Q-network to target network."""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def end_episode(self) -> None:
        """Called at the end of each episode."""
        self.episode_count += 1
        self.decay_epsilon()
        
        # Flush n-step buffer
        if self.use_n_step:
            self.memory.reset_episode()
        
        # Update target network periodically
        if self.episode_count % self.target_update_freq == 0:
            self.update_target_network()
    
    def save(self, path: str) -> None:
        """Save model to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Normalize path - use .h5 extension
        base_path = path.replace('.pt', '').replace('.h5', '')
        
        # Save weights
        self.q_network.save_weights(f"{base_path}.h5")
        self.target_network.save_weights(f"{base_path}_target.h5")
        
        # Save metadata
        import json
        metadata = {
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'training_step': self.training_step,
            'use_dueling': hasattr(self.q_network, 'use_dueling')  # Track architecture
        }
        with open(f"{base_path}_meta.json", 'w') as f:
            json.dump(metadata, f)
        
        print(f"Model saved to {base_path}.h5")
    
    def load(self, path: str) -> None:
        """Load model from file."""
        # Normalize path
        base_path = path.replace('.pt', '').replace('.h5', '').replace('_q.weights', '').replace('_meta', '')
        
        # Try new format first, then old format
        q_path = f"{base_path}.h5"
        target_path = f"{base_path}_target.h5"
        meta_path = f"{base_path}_meta.json"
        
        # Fall back to old format if needed
        if not os.path.exists(q_path):
            q_path = f"{base_path}_q.weights.h5"
            target_path = f"{base_path}_target.weights.h5"
        
        if os.path.exists(q_path):
            self.q_network.load_weights(q_path)
            if os.path.exists(target_path):
                self.target_network.load_weights(target_path)
            print(f"Model loaded from {q_path}")
        else:
            print(f"Warning: Model file not found at {q_path}")
        
        # Load metadata
        import json
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            self.epsilon = metadata.get('epsilon', 0.01)
            self.episode_count = metadata.get('episode_count', 0)
            self.training_step = metadata.get('training_step', 0)
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions."""
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        return self.q_network(state_tensor, training=False).numpy()[0]


class BaselineAgent:
    """
    Collection of baseline agents for comparison.
    
    Includes:
    - Random agent (lower bound)
    - Simple threshold agent
    - Snort-inspired rule-based agent
    - NIST 800-61 incident response guidelines
    - MITRE ATT&CK pattern matching
    - Adaptive moving average agent
    """
    
    @staticmethod
    def random_agent(action_size: int) -> int:
        """Random action selection."""
        return random.randrange(action_size)
    
    @staticmethod
    def threshold_agent(state: np.ndarray, thresholds: dict = None) -> int:
        """Simple threshold-based agent."""
        if thresholds is None:
            thresholds = {
                'login_high': 30,
                'login_critical': 50,
                'file_high': 50,
                'file_critical': 100,
                'cpu_high': 70,
                'delta_threshold': 20
            }
        
        login_rate = state[0]
        file_rate = state[1]
        cpu_usage = state[2]
        
        if len(state) >= 10:
            login_delta = state[3]
            file_delta = state[4]
            sustained = state[8]
            
            if login_delta > thresholds['delta_threshold'] or file_delta > thresholds['delta_threshold'] * 2:
                return 4
            
            if sustained > 0.7:
                return 3
        
        if login_rate > thresholds['login_critical'] and file_rate > thresholds['file_critical']:
            return 4
        
        if login_rate > thresholds['login_critical']:
            return 2
        elif login_rate > thresholds['login_high']:
            return 1
        
        if file_rate > thresholds['file_critical'] and cpu_usage > thresholds['cpu_high']:
            return 3
        elif file_rate > thresholds['file_high']:
            return 3
        
        return 0
    
    @staticmethod
    def snort_inspired_agent(state: np.ndarray) -> int:
        """
        Snort-inspired rule-based detection.
        
        Based on Snort IDS rule thresholds and detection logic:
        - SID 1000001: SSH brute force (>10 attempts/min)
        - SID 1000015: Potential ransomware (>50 file ops/min + high CPU)
        - Uses counting and threshold methods from Snort
        
        Reference: Snort Users Manual, Section 3.7 - Thresholding
        """
        login_rate = state[0]
        file_rate = state[1]
        cpu_usage = state[2]
        
        # Snort-style thresholds (based on common rule configurations)
        SSH_BRUTEFORCE_THRESHOLD = 10      # SID:1000001
        SSH_BRUTEFORCE_CRITICAL = 30       # Escalation threshold
        RAPID_FILE_ACCESS = 75             # Ransomware indicator
        CPU_ANOMALY = 85                   # System stress indicator
        
        alert_level = 0  # No alert
        
        # Rule 1: SSH Brute Force Detection (similar to SID:1000001)
        # threshold type limit, track by_src, count 10, seconds 60
        if login_rate > SSH_BRUTEFORCE_CRITICAL:
            alert_level = max(alert_level, 3)  # High priority
        elif login_rate > SSH_BRUTEFORCE_THRESHOLD:
            alert_level = max(alert_level, 2)  # Medium priority
        
        # Rule 2: Ransomware Behavior (file encryption pattern)
        # alert tcp any any -> any any (msg:"Possible ransomware activity"; 
        # flow:established; content:"encrypted"; threshold:type both,track by_src,count 50,seconds 60;)
        if file_rate > RAPID_FILE_ACCESS and cpu_usage > CPU_ANOMALY:
            alert_level = max(alert_level, 4)  # Critical - isolate immediately
        elif file_rate > RAPID_FILE_ACCESS:
            alert_level = max(alert_level, 3)  # High priority
        
        # Rule 3: Combined attack indicators
        if login_rate > SSH_BRUTEFORCE_THRESHOLD and file_rate > RAPID_FILE_ACCESS / 2:
            alert_level = max(alert_level, 4)  # Multi-vector attack
        
        # Map alert level to action
        # 0: do_nothing, 1: block_ip, 2: lock_account, 3: terminate_process, 4: isolate_host
        return min(alert_level, 4)
    
    @staticmethod
    def nist_incident_response_agent(state: np.ndarray, history: list = None) -> int:
        """
        NIST SP 800-61 Rev. 2 based incident response.
        
        Implements the NIST incident response lifecycle:
        1. Detection - Identify potential incidents
        2. Analysis - Determine severity
        3. Containment - Choose appropriate response
        4. Eradication - Remove threat
        
        Reference: NIST SP 800-61 Rev. 2, Section 3.2
        """
        login_rate = state[0]
        file_rate = state[1]
        cpu_usage = state[2]
        
        # NIST severity levels (based on Table 3-2 in SP 800-61)
        LOW_THRESHOLD = 20
        MEDIUM_THRESHOLD = 40
        HIGH_THRESHOLD = 60
        CRITICAL_THRESHOLD = 80
        
        # Calculate functional impact (Section 3.2.6)
        login_impact = min(login_rate / CRITICAL_THRESHOLD, 1.0)
        file_impact = min(file_rate / (CRITICAL_THRESHOLD * 2), 1.0)
        cpu_impact = min(cpu_usage / 100, 1.0)
        
        # Information impact score (weighted)
        impact_score = 0.4 * login_impact + 0.4 * file_impact + 0.2 * cpu_impact
        
        # NIST response based on impact (Table 3-3)
        if impact_score > 0.8:
            # High impact: Immediate containment required
            return 4  # Isolate host
        elif impact_score > 0.6:
            # Medium-High: Aggressive containment
            return 3  # Terminate process
        elif impact_score > 0.4:
            # Medium: Standard containment
            return 2  # Lock account
        elif impact_score > 0.2:
            # Low: Minimal response
            return 1  # Block IP
        else:
            # Negligible: Continue monitoring
            return 0  # Do nothing
    
    @staticmethod
    def mitre_attack_agent(state: np.ndarray) -> int:
        """
        MITRE ATT&CK framework pattern matching.
        
        Detects attack patterns based on MITRE ATT&CK tactics:
        - T1110: Brute Force (Credential Access)
        - T1486: Data Encrypted for Impact (Ransomware)
        - T1496: Resource Hijacking (Cryptomining - high CPU)
        
        Reference: https://attack.mitre.org/
        """
        login_rate = state[0]
        file_rate = state[1]
        cpu_usage = state[2]
        
        detected_techniques = []
        
        # T1110 - Brute Force: Credential Access
        # Sub-techniques: T1110.001 (Password Guessing), T1110.003 (Password Spraying)
        if login_rate > 25:
            detected_techniques.append(('T1110', 'brute_force', 2))  # (technique, name, severity)
        if login_rate > 50:
            detected_techniques.append(('T1110.001', 'password_guessing', 3))
        
        # T1486 - Data Encrypted for Impact (Ransomware)
        if file_rate > 80 and cpu_usage > 60:
            detected_techniques.append(('T1486', 'ransomware', 4))
        elif file_rate > 100:
            detected_techniques.append(('T1486', 'potential_ransomware', 3))
        
        # T1496 - Resource Hijacking (Cryptomining)
        if cpu_usage > 90 and file_rate < 30:
            detected_techniques.append(('T1496', 'cryptomining', 2))
        
        # T1071 - Application Layer Protocol (C2 communication)
        # Indicated by sustained unusual activity
        if len(state) >= 10 and state[8] > 0.5:  # sustained_indicator
            detected_techniques.append(('T1071', 'c2_communication', 3))
        
        # Response based on highest severity technique detected
        if not detected_techniques:
            return 0
        
        max_severity = max(t[2] for t in detected_techniques)
        return min(max_severity, 4)
    
    @staticmethod
    def adaptive_moving_average_agent(state: np.ndarray, history: deque = None) -> int:
        """
        Adaptive agent using moving average and standard deviation.
        
        This is a more sophisticated baseline that:
        1. Maintains historical averages
        2. Detects anomalies based on z-score
        3. Adapts thresholds over time
        
        Similar to statistical anomaly detection in modern SIEM systems.
        """
        if history is None:
            # First call - use static thresholds
            return BaselineAgent.threshold_agent(state)
        
        login_rate = state[0]
        file_rate = state[1]
        cpu_usage = state[2]
        
        # Calculate z-scores from history
        if len(history) < 10:
            return BaselineAgent.threshold_agent(state)
        
        hist_array = np.array(list(history))
        means = np.mean(hist_array, axis=0)
        stds = np.std(hist_array, axis=0) + 1e-6  # Avoid division by zero
        
        login_zscore = (login_rate - means[0]) / stds[0]
        file_zscore = (file_rate - means[1]) / stds[1]
        cpu_zscore = (cpu_usage - means[2]) / stds[2]
        
        # Anomaly detection based on z-score
        max_zscore = max(login_zscore, file_zscore, cpu_zscore)
        
        if max_zscore > 4.0:
            return 4  # Extreme anomaly - isolate
        elif max_zscore > 3.0:
            return 3  # Major anomaly - terminate
        elif max_zscore > 2.5:
            return 2  # Significant anomaly - lock
        elif max_zscore > 2.0:
            return 1  # Minor anomaly - block
        else:
            return 0  # Normal
    
    @staticmethod
    def always_action_agent(action: int) -> int:
        """Always take the same action."""
        return action


class RuleBasedAgentWrapper:
    """
    Wrapper class to maintain state for stateful baseline agents.
    """
    
    def __init__(self, agent_type: str, window_size: int = 50):
        """
        Initialize wrapper.
        
        Args:
            agent_type: Type of agent ('snort', 'nist', 'mitre', 'adaptive')
            window_size: History window for adaptive agents
        """
        self.agent_type = agent_type
        self.history = deque(maxlen=window_size)
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action and update history."""
        self.history.append(state[:3])  # Store login, file, cpu
        
        if self.agent_type == 'snort':
            return BaselineAgent.snort_inspired_agent(state)
        elif self.agent_type == 'nist':
            return BaselineAgent.nist_incident_response_agent(state)
        elif self.agent_type == 'mitre':
            return BaselineAgent.mitre_attack_agent(state)
        elif self.agent_type == 'adaptive':
            return BaselineAgent.adaptive_moving_average_agent(state, self.history)
        else:
            return BaselineAgent.threshold_agent(state)
    
    def reset(self):
        """Reset history for new episode."""
        self.history.clear()


if __name__ == "__main__":
    # Test the DQN agent
    print("Testing TensorFlow DQN Agent")
    print("=" * 50)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    
    # Test standard DQN
    print("\n--- Testing Standard DQN ---")
    agent = DQNAgent(state_size=10, action_size=5)
    
    for i in range(100):
        state = np.random.randn(10).astype(np.float32)
        action = agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(10).astype(np.float32)
        done = i % 20 == 19
        
        agent.store_experience(state, action, reward, next_state, done)
        loss = agent.train_step()
        
        if done:
            agent.end_episode()
    
    print(f"Training completed. Episodes: {agent.episode_count}")
    
    # Test N-step DQN
    print("\n--- Testing N-step DQN ---")
    agent_nstep = DQNAgent(state_size=10, action_size=5, use_n_step=True, n_steps=3)
    
    for i in range(100):
        state = np.random.randn(10).astype(np.float32)
        action = agent_nstep.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(10).astype(np.float32)
        done = i % 20 == 19
        
        agent_nstep.store_experience(state, action, reward, next_state, done)
        loss = agent_nstep.train_step()
        
        if done:
            agent_nstep.end_episode()
    
    print(f"N-step training completed. Episodes: {agent_nstep.episode_count}")
    
    # Test save/load
    agent.save("models/test_agent.pt")
    agent.load("models/test_agent.pt")
    
    print("\nAgent test complete!")
