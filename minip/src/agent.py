"""
DQN Agent for Automated Incident Response.

This module implements:
- Deep Q-Network (DQN) with experience replay
- Double DQN for reduced overestimation
- N-step returns for better temporal credit assignment
- Epsilon-greedy exploration with decay
- Target network for stable training
- Prioritized experience replay (optional)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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


class QNetwork(nn.Module):
    """
    Neural network for Q-value approximation.
    
    Architecture:
    - Input: State observation
    - Hidden layers with ReLU activation
    - Output: Q-values for each action
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_layers: List[int] = [128, 64, 32],
        dropout_rate: float = 0.1,
        use_dueling: bool = False
    ):
        """
        Initialize Q-Network.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            hidden_layers: Sizes of hidden layers
            dropout_rate: Dropout rate for regularization
            use_dueling: Whether to use dueling architecture
        """
        super(QNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.use_dueling = use_dueling
        
        # Build shared layers
        layers = []
        prev_size = state_size
        
        for i, hidden_size in enumerate(hidden_layers[:-1]):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        self.shared_layers = nn.Sequential(*layers)
        
        if use_dueling:
            # Dueling architecture: separate value and advantage streams
            last_hidden = hidden_layers[-1] if hidden_layers else state_size
            
            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(prev_size, last_hidden),
                nn.ReLU(),
                nn.Linear(last_hidden, 1)
            )
            
            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(prev_size, last_hidden),
                nn.ReLU(),
                nn.Linear(last_hidden, action_size)
            )
        else:
            # Standard architecture
            self.output_layers = nn.Sequential(
                nn.Linear(prev_size, hidden_layers[-1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_layers[-1], action_size)
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State observation tensor
            
        Returns:
            Q-values for each action
        """
        x = self.shared_layers(state)
        
        if self.use_dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            # Q = V + (A - mean(A))
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
            return q_values
        else:
            return self.output_layers(x)


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
    
    Accumulates n-step returns before storing experiences.
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
        
        # Temporary storage for n-step calculation
        self.n_step_buffer = deque(maxlen=n_steps)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add experience and compute n-step returns when ready.
        """
        self.n_step_buffer.append(Experience(state, action, reward, next_state, done))
        
        # Only store when we have enough steps or episode ends
        if len(self.n_step_buffer) == self.n_steps or done:
            # Calculate n-step return
            n_step_reward = 0.0
            for i, exp in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * exp.reward
                if exp.done:
                    break
            
            # Store the n-step experience
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
            
            # Clear buffer if episode ended
            if done:
                self.n_step_buffer.clear()
    
    def sample(self, batch_size: int) -> List[NStepExperience]:
        """Sample random batch of n-step experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def reset_episode(self) -> None:
        """Reset n-step buffer at episode end."""
        # Flush remaining experiences
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
    
    Samples experiences with probability proportional to their TD error.
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        """
        Initialize prioritized buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
        """
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
        """
        Sample batch with prioritized sampling.
        
        Args:
            batch_size: Number of experiences
            beta: Importance sampling exponent
            
        Returns:
            Tuple of (experiences, importance weights, indices)
        """
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
    Deep Q-Network Agent for incident response.
    
    Features:
    - Experience replay for stable training
    - Target network for reducing overestimation
    - Double DQN for action selection
    - N-step returns for better credit assignment
    - Epsilon-greedy exploration with decay
    - Support for prioritized experience replay
    - Optional dueling architecture
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
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Q-Networks
        self.q_network = QNetwork(
            state_size, action_size, 
            config.hidden_layers,
            use_dueling=use_dueling
        ).to(self.device)
        
        self.target_network = QNetwork(
            state_size, action_size,
            config.hidden_layers,
            use_dueling=use_dueling
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate
        )
        
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
            # Random exploration
            return random.randrange(self.action_size)
        
        # Greedy action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
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
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            experiences = self.memory.sample(self.batch_size)
            weights = torch.ones(len(experiences)).to(self.device)
            indices = None
        
        # Handle n-step experiences differently
        if self.use_n_step:
            states = torch.FloatTensor(
                np.array([e.state for e in experiences])
            ).to(self.device)
            
            actions = torch.LongTensor(
                [e.action for e in experiences]
            ).to(self.device)
            
            rewards = torch.FloatTensor(
                [e.n_step_reward for e in experiences]
            ).to(self.device)
            
            next_states = torch.FloatTensor(
                np.array([e.next_state for e in experiences])
            ).to(self.device)
            
            dones = torch.FloatTensor(
                [float(e.done) for e in experiences]
            ).to(self.device)
            
            n_values = torch.FloatTensor(
                [e.n for e in experiences]
            ).to(self.device)
            
            # Compute discount for n-step
            gamma_n = self.gamma ** n_values
        else:
            states = torch.FloatTensor(
                np.array([e.state for e in experiences])
            ).to(self.device)
            
            actions = torch.LongTensor(
                [e.action for e in experiences]
            ).to(self.device)
            
            rewards = torch.FloatTensor(
                [e.reward for e in experiences]
            ).to(self.device)
            
            next_states = torch.FloatTensor(
                np.array([e.next_state for e in experiences])
            ).to(self.device)
            
            dones = torch.FloatTensor(
                [float(e.done) for e in experiences]
            ).to(self.device)
            
            gamma_n = self.gamma
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values (Double DQN)
        with torch.no_grad():
            # Select best action using online network
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            # Evaluate using target network
            next_q = self.target_network(next_states).gather(1, next_actions)
            
            if self.use_n_step:
                target_q = rewards.unsqueeze(1) + gamma_n.unsqueeze(1) * next_q * (1 - dones.unsqueeze(1))
            else:
                target_q = rewards.unsqueeze(1) + gamma_n * next_q * (1 - dones.unsqueeze(1))
        
        # TD errors for prioritized replay
        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
        
        # Weighted loss
        loss = (weights.unsqueeze(1) * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities
        if self.use_prioritized and indices is not None:
            self.memory.update_priorities(indices, td_errors.flatten() + 1e-6)
        
        self.training_step += 1
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def update_target_network(self) -> None:
        """Copy weights from Q-network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
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
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'training_step': self.training_step
        }, path)
        
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']
        self.training_step = checkpoint['training_step']
        
        print(f"Model loaded from {path}")
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_network(state_tensor).cpu().numpy()[0]


class BaselineAgent:
    """
    Simple baseline agents for comparison.
    """
    
    @staticmethod
    def random_agent(action_size: int) -> int:
        """Random action selection."""
        return random.randrange(action_size)
    
    @staticmethod
    def threshold_agent(state: np.ndarray, thresholds: dict = None) -> int:
        """
        Rule-based agent using thresholds.
        
        Args:
            state: Observation (supports both 4D and 10D)
            thresholds: Dict of threshold values
            
        Returns:
            Action index
        """
        if thresholds is None:
            thresholds = {
                'login_high': 30,
                'login_critical': 50,
                'file_high': 50,
                'file_critical': 100,
                'cpu_high': 70,
                'delta_threshold': 20  # For rate of change
            }
        
        # Extract features (works for both 4D and 10D states)
        login_rate = state[0]
        file_rate = state[1]
        cpu_usage = state[2]
        
        # If enhanced features available, use rate of change
        if len(state) >= 10:
            login_delta = state[3]
            file_delta = state[4]
            sustained = state[8]
            
            # Rapid increase in activity
            if login_delta > thresholds['delta_threshold'] or file_delta > thresholds['delta_threshold'] * 2:
                return 4  # Isolate host (aggressive response to rapid escalation)
            
            # Sustained high activity
            if sustained > 0.7:
                return 3  # Terminate process
        
        # Critical situation: high login + high file access
        if login_rate > thresholds['login_critical'] and file_rate > thresholds['file_critical']:
            return 4  # Isolate host
        
        # High login attempts
        if login_rate > thresholds['login_critical']:
            return 2  # Lock account
        elif login_rate > thresholds['login_high']:
            return 1  # Block IP
        
        # High file access with high CPU (ransomware-like)
        if file_rate > thresholds['file_critical'] and cpu_usage > thresholds['cpu_high']:
            return 3  # Terminate process
        elif file_rate > thresholds['file_high']:
            return 3  # Terminate process
        
        return 0  # Do nothing
    
    @staticmethod
    def always_action_agent(action: int) -> int:
        """Always take the same action (for comparison)."""
        return action


if __name__ == "__main__":
    # Test the DQN agent
    print("Testing DQN Agent")
    print("=" * 50)
    
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
