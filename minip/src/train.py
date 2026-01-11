"""
Training Script for Automated Incident Response RL Agent.

This script handles:
- Training loop with experience collection
- Model checkpointing
- Training metrics logging
- Statistical significance testing
- Support for enhanced features (N-step, dueling)
"""

import numpy as np
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from config import get_config, Config
from incident_env import IncidentResponseEnv
from agent import DQNAgent, BaselineAgent


class TrainingMetrics:
    """Tracks and stores training metrics."""
    
    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_losses: List[float] = []
        self.epsilons: List[float] = []
        self.attacks_contained: List[int] = []
        self.false_positives: List[int] = []
        self.data_loss_events: List[int] = []
        
    def add_episode(
        self,
        reward: float,
        length: int,
        avg_loss: float,
        epsilon: float,
        stats: Dict
    ) -> None:
        """Add metrics for one episode."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_losses.append(avg_loss)
        self.epsilons.append(epsilon)
        self.attacks_contained.append(stats.get('attacks_contained', 0))
        self.false_positives.append(stats.get('false_positives', 0))
        self.data_loss_events.append(stats.get('data_loss_events', 0))
    
    def get_recent_average(self, metric: str, window: int = 100) -> float:
        """Get moving average of a metric."""
        values = getattr(self, metric)
        if len(values) < window:
            return np.mean(values) if values else 0.0
        return np.mean(values[-window:])
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_losses': self.episode_losses,
            'epsilons': self.epsilons,
            'attacks_contained': self.attacks_contained,
            'false_positives': self.false_positives,
            'data_loss_events': self.data_loss_events
        }
    
    def save(self, path: str) -> None:
        """Save metrics to JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class Trainer:
    """
    Trainer class for DQN agent.
    
    Handles the complete training loop including:
    - Environment interaction
    - Experience collection
    - Agent updates
    - Logging and checkpointing
    - Statistical significance testing
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        attack_type: str = "random",
        use_prioritized_replay: bool = False,
        use_n_step: bool = False,
        n_steps: int = 3,
        use_dueling: bool = False,
        use_enhanced_features: bool = True,
        checkpoint_dir: str = "models",
        log_dir: str = "logs"
    ):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
            attack_type: Type of attack to train on
            use_prioritized_replay: Use prioritized experience replay
            use_n_step: Use N-step returns for better credit assignment
            n_steps: Number of steps for N-step returns
            use_dueling: Use dueling network architecture
            use_enhanced_features: Use enhanced 10D observation space
            checkpoint_dir: Directory for model checkpoints
            log_dir: Directory for training logs
        """
        self.config = config if config else get_config()
        self.attack_type = attack_type
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.use_enhanced_features = use_enhanced_features
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize environment
        self.env = IncidentResponseEnv(
            config=self.config,
            attack_type=attack_type,
            use_enhanced_features=use_enhanced_features
        )
        
        # Initialize agent
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        
        self.agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            config=self.config.agent,
            use_prioritized_replay=use_prioritized_replay,
            use_n_step=use_n_step,
            n_steps=n_steps,
            use_dueling=use_dueling
        )
        
        # Training metrics
        self.metrics = TrainingMetrics()
        
        # Store rewards for statistical testing
        self.all_rewards = []
        
        # Best model tracking
        self.best_reward = float('-inf')
        
        print(f"\n{'='*60}")
        print("Incident Response RL Training Initialized")
        print(f"{'='*60}")
        print(f"State size: {state_size}")
        print(f"Action size: {action_size}")
        print(f"Attack type: {attack_type}")
        print(f"Enhanced features: {use_enhanced_features}")
        print(f"N-step returns: {use_n_step} (n={n_steps})" if use_n_step else "N-step returns: disabled")
        print(f"Dueling architecture: {use_dueling}")
        print(f"Device: {self.agent.device}")
        print(f"{'='*60}\n")
    
    def train_episode(self) -> Tuple[float, int, float, Dict]:
        """
        Train for one episode.
        
        Returns:
            Tuple of (total_reward, steps, avg_loss, episode_stats)
        """
        state, info = self.env.reset()
        total_reward = 0.0
        episode_losses = []
        step = 0
        
        while True:
            # Select action
            action = self.agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store experience
            self.agent.store_experience(state, action, reward, next_state, done)
            
            # Train agent
            loss = self.agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            
            total_reward += reward
            state = next_state
            step += 1
            
            if done:
                break
        
        # End of episode
        self.agent.end_episode()
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        
        return total_reward, step, avg_loss, info.get('episode_stats', {})
    
    def train(
        self,
        num_episodes: Optional[int] = None,
        eval_frequency: int = 50,
        save_frequency: int = 100,
        verbose: bool = True
    ) -> TrainingMetrics:
        """
        Run full training loop.
        
        Args:
            num_episodes: Number of episodes to train
            eval_frequency: Evaluate every N episodes
            save_frequency: Save checkpoint every N episodes
            verbose: Print progress
            
        Returns:
            Training metrics
        """
        if num_episodes is None:
            num_episodes = self.config.agent.num_episodes
        
        if verbose:
            print(f"Starting training for {num_episodes} episodes...")
        start_time = time.time()
        
        progress_bar = tqdm(range(num_episodes), desc="Training", disable=not verbose)
        
        for episode in progress_bar:
            # Train episode
            reward, steps, avg_loss, stats = self.train_episode()
            
            # Store reward for statistical testing
            self.all_rewards.append(reward)
            
            # Record metrics
            self.metrics.add_episode(
                reward=reward,
                length=steps,
                avg_loss=avg_loss,
                epsilon=self.agent.epsilon,
                stats=stats
            )
            
            # Update progress bar
            if verbose:
                avg_reward = self.metrics.get_recent_average('episode_rewards')
                progress_bar.set_postfix({
                    'reward': f'{reward:.1f}',
                    'avg': f'{avg_reward:.1f}',
                    'eps': f'{self.agent.epsilon:.3f}'
                })
            
            # Save checkpoint
            if (episode + 1) % save_frequency == 0:
                self._save_checkpoint(episode + 1)
            
            # Evaluation
            if (episode + 1) % eval_frequency == 0 and verbose:
                self._evaluate_and_log(episode + 1)
            
            # Save best model
            if reward > self.best_reward:
                self.best_reward = reward
                self.agent.save(os.path.join(self.checkpoint_dir, 'best_model.pt'))
        
        # Training complete
        total_time = time.time() - start_time
        if verbose:
            print(f"\nTraining complete in {total_time/60:.1f} minutes")
            print(f"Best episode reward: {self.best_reward:.2f}")
        
        # Save final model and metrics
        self.agent.save(os.path.join(self.checkpoint_dir, 'final_model.pt'))
        self.metrics.save(os.path.join(self.log_dir, 'training_metrics.json'))
        
        return self.metrics
    
    def _save_checkpoint(self, episode: int) -> None:
        """Save training checkpoint."""
        path = os.path.join(self.checkpoint_dir, f'checkpoint_ep{episode}.pt')
        self.agent.save(path)
    
    def _evaluate_and_log(self, episode: int) -> Dict:
        """Evaluate agent and log results."""
        eval_results = self.evaluate(num_episodes=10)
        
        print(f"\n--- Evaluation at Episode {episode} ---")
        print(f"Average Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"Success Rate: {eval_results['success_rate']*100:.1f}%")
        print(f"Avg False Positives: {eval_results['avg_false_positives']:.2f}")
        print(f"Epsilon: {self.agent.epsilon:.4f}")
        
        return eval_results
    
    def evaluate(
        self,
        num_episodes: int = 100,
        render: bool = False
    ) -> Dict:
        """
        Evaluate trained agent.
        
        Args:
            num_episodes: Number of evaluation episodes
            render: Whether to render episodes
            
        Returns:
            Evaluation results dictionary
        """
        rewards = []
        successes = 0
        false_positives_list = []
        contained_list = []
        
        for _ in range(num_episodes):
            state, info = self.env.reset()
            episode_reward = 0.0
            
            while True:
                action = self.agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                state = next_state
                
                if render:
                    self.env.render()
                
                if terminated or truncated:
                    break
            
            rewards.append(episode_reward)
            stats = info.get('episode_stats', {})
            
            # Success = contained attack or no attack occurred without false positives
            if stats.get('attacks_contained', 0) > 0 or (
                stats.get('missed_attacks', 0) == 0 and 
                stats.get('false_positives', 0) == 0
            ):
                successes += 1
            
            false_positives_list.append(stats.get('false_positives', 0))
            contained_list.append(stats.get('attacks_contained', 0))
        
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'rewards': rewards,  # Store full array for statistical testing
            'success_rate': successes / num_episodes,
            'avg_false_positives': np.mean(false_positives_list),
            'avg_contained': np.mean(contained_list)
        }
    
    def compare_with_baselines(
        self,
        num_episodes: int = 100,
        include_statistical_tests: bool = True
    ) -> Dict:
        """
        Compare trained agent with baseline policies.
        
        Args:
            num_episodes: Number of episodes for comparison
            include_statistical_tests: Whether to perform statistical significance tests
            
        Returns:
            Comparison results with optional statistical analysis
        """
        results = {}
        reward_arrays = {}
        
        # Trained DQN agent
        print("Evaluating DQN Agent...")
        dqn_eval = self.evaluate(num_episodes)
        results['dqn'] = dqn_eval
        reward_arrays['dqn'] = dqn_eval['rewards']
        
        # Random agent
        print("Evaluating Random Agent...")
        random_rewards = []
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0
            while True:
                action = BaselineAgent.random_agent(self.env.action_space.n)
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            random_rewards.append(episode_reward)
        results['random'] = {'avg_reward': np.mean(random_rewards), 'std_reward': np.std(random_rewards)}
        reward_arrays['random'] = random_rewards
        
        # Threshold-based agent
        print("Evaluating Threshold Agent...")
        threshold_rewards = []
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0
            while True:
                action = BaselineAgent.threshold_agent(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            threshold_rewards.append(episode_reward)
        results['threshold'] = {'avg_reward': np.mean(threshold_rewards), 'std_reward': np.std(threshold_rewards)}
        reward_arrays['threshold'] = threshold_rewards
        
        # Snort-inspired agent (based on real IDS rules)
        print("Evaluating Snort-Inspired Agent...")
        snort_rewards = []
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0
            while True:
                action = BaselineAgent.snort_inspired_agent(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            snort_rewards.append(episode_reward)
        results['snort'] = {'avg_reward': np.mean(snort_rewards), 'std_reward': np.std(snort_rewards)}
        reward_arrays['snort'] = snort_rewards
        
        # NIST 800-61 incident response agent
        print("Evaluating NIST 800-61 Agent...")
        nist_rewards = []
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0
            while True:
                action = BaselineAgent.nist_incident_response_agent(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            nist_rewards.append(episode_reward)
        results['nist_800_61'] = {'avg_reward': np.mean(nist_rewards), 'std_reward': np.std(nist_rewards)}
        reward_arrays['nist_800_61'] = nist_rewards
        
        # MITRE ATT&CK pattern matching agent
        print("Evaluating MITRE ATT&CK Agent...")
        mitre_rewards = []
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0
            while True:
                action = BaselineAgent.mitre_attack_agent(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            mitre_rewards.append(episode_reward)
        results['mitre_attack'] = {'avg_reward': np.mean(mitre_rewards), 'std_reward': np.std(mitre_rewards)}
        reward_arrays['mitre_attack'] = mitre_rewards
        
        # Do-nothing agent
        print("Evaluating Do-Nothing Agent...")
        nothing_rewards = []
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0
            while True:
                state, reward, terminated, truncated, _ = self.env.step(0)
                episode_reward += reward
                if terminated or truncated:
                    break
            nothing_rewards.append(episode_reward)
        results['do_nothing'] = {'avg_reward': np.mean(nothing_rewards), 'std_reward': np.std(nothing_rewards)}
        reward_arrays['do_nothing'] = nothing_rewards
        
        # Print comparison
        print("\n" + "="*70)
        print("Agent Comparison Results")
        print("="*70)
        print(f"{'Agent':<20} {'Avg Reward':<15} {'Std':<10}")
        print("-"*45)
        for agent_name, agent_results in results.items():
            avg = agent_results.get('avg_reward', 0)
            std = agent_results.get('std_reward', 0)
            print(f"{agent_name:<20} {avg:>10.2f}      {std:>6.2f}")
        print("="*70)
        
        # Statistical significance testing
        if include_statistical_tests:
            print("\n" + "-"*70)
            print("Statistical Significance Tests (DQN vs Baselines)")
            print("-"*70)
            
            try:
                from evaluate import StatisticalAnalyzer
                analyzer = StatisticalAnalyzer()
                
                dqn_rewards = np.array(reward_arrays['dqn'])
                
                for baseline_name, baseline_rewards in reward_arrays.items():
                    if baseline_name == 'dqn':
                        continue
                    
                    baseline_rewards = np.array(baseline_rewards)
                    test_result = analyzer.independent_t_test(dqn_rewards, baseline_rewards)
                    
                    sig_marker = "✓" if test_result.is_significant else "✗"
                    print(f"\nDQN vs {baseline_name}:")
                    print(f"  T-statistic: {test_result.statistic:.4f}")
                    print(f"  P-value: {test_result.p_value:.6f}")
                    print(f"  Cohen's d: {test_result.effect_size:.4f}")
                    print(f"  95% CI: ({test_result.confidence_interval[0]:.2f}, {test_result.confidence_interval[1]:.2f})")
                    print(f"  Significant: {sig_marker}")
                    
                    # Store in results
                    results[f'{baseline_name}_test'] = {
                        'statistic': test_result.statistic,
                        'p_value': test_result.p_value,
                        'effect_size': test_result.effect_size,
                        'is_significant': test_result.is_significant
                    }
            except ImportError:
                print("  (Statistical tests skipped - scipy not available)")
            
            print("-"*70)
        
        return results


def train_agent(
    num_episodes: int = 1000,
    attack_type: str = "random",
    checkpoint_dir: str = "models",
    **kwargs
) -> Trainer:
    """
    Convenience function to train an agent.
    
    Args:
        num_episodes: Number of training episodes
        attack_type: Attack type to train on
        checkpoint_dir: Directory for checkpoints
        **kwargs: Additional trainer arguments
        
    Returns:
        Trainer object with trained agent
    """
    trainer = Trainer(
        attack_type=attack_type,
        checkpoint_dir=checkpoint_dir,
        **kwargs
    )
    
    trainer.train(num_episodes=num_episodes)
    
    return trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Incident Response RL Agent")
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes')
    parser.add_argument('--attack', type=str, default='random', 
                       choices=['bruteforce', 'ransomware', 'both', 'random'],
                       help='Attack type to train on')
    parser.add_argument('--checkpoint-dir', type=str, default='models')
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--n-step', action='store_true', help='Use N-step returns')
    parser.add_argument('--n-steps', type=int, default=3, help='N for N-step returns')
    parser.add_argument('--dueling', action='store_true', help='Use dueling architecture')
    parser.add_argument('--no-enhanced', action='store_true', help='Disable enhanced features (use 4D state)')
    parser.add_argument('--eval', action='store_true', help='Run evaluation after training')
    parser.add_argument('--compare', action='store_true', help='Compare with baselines')
    
    args = parser.parse_args()
    
    # Create trainer and train
    trainer = Trainer(
        attack_type=args.attack,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        use_n_step=args.n_step,
        n_steps=args.n_steps,
        use_dueling=args.dueling,
        use_enhanced_features=not args.no_enhanced
    )
    
    # Train
    trainer.train(num_episodes=args.episodes)
    
    # Evaluation
    if args.eval:
        print("\nRunning final evaluation...")
        eval_results = trainer.evaluate(num_episodes=100)
        print(f"\nFinal Evaluation Results:")
        print(f"Average Reward: {eval_results['avg_reward']:.2f}")
        print(f"Success Rate: {eval_results['success_rate']*100:.1f}%")
    
    # Baseline comparison with statistical tests
    if args.compare:
        print("\nComparing with baseline agents...")
        comparison = trainer.compare_with_baselines(num_episodes=100, include_statistical_tests=True)
