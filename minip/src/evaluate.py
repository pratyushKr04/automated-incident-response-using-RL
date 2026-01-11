"""
Evaluation and Visualization Module for Automated Incident Response.

This module provides:
- Model evaluation utilities
- Statistical significance testing (t-tests, confidence intervals)
- Training visualizations (learning curves, rewards)
- Policy analysis (Q-value distributions, action preferences)
- Hyperparameter sensitivity analysis
- Performance comparison plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
from scipy import stats
from dataclasses import dataclass
import warnings

# Set style for plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")


@dataclass
class StatisticalTestResult:
    """Results from statistical significance testing."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    
    def __str__(self) -> str:
        sig = "✓ Significant" if self.is_significant else "✗ Not significant"
        return (f"{self.test_name}: stat={self.statistic:.4f}, p={self.p_value:.4f}, "
                f"effect_size={self.effect_size:.4f}, CI={self.confidence_interval}, {sig}")


class StatisticalAnalyzer:
    """
    Performs statistical significance testing for RL evaluation.
    
    Implements:
    - Independent samples t-test
    - Welch's t-test
    - Cohen's d effect size
    - Bootstrap confidence intervals
    - Mann-Whitney U test (non-parametric alternative)
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize analyzer.
        
        Args:
            alpha: Significance level for hypothesis testing
        """
        self.alpha = alpha
    
    def independent_t_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        equal_var: bool = False
    ) -> StatisticalTestResult:
        """
        Perform independent samples t-test (Welch's by default).
        
        Args:
            group1: First group's samples (e.g., DQN rewards)
            group2: Second group's samples (e.g., baseline rewards)
            equal_var: Whether to assume equal variances
            
        Returns:
            StatisticalTestResult object
        """
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
        
        # Cohen's d effect size
        pooled_std = np.sqrt(((len(group1)-1)*np.var(group1, ddof=1) + 
                              (len(group2)-1)*np.var(group2, ddof=1)) / 
                             (len(group1) + len(group2) - 2))
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval for difference in means
        diff_mean = np.mean(group1) - np.mean(group2)
        se_diff = np.sqrt(np.var(group1, ddof=1)/len(group1) + 
                         np.var(group2, ddof=1)/len(group2))
        ci = stats.t.interval(1-self.alpha, df=len(group1)+len(group2)-2, 
                             loc=diff_mean, scale=se_diff)
        
        return StatisticalTestResult(
            test_name="Welch's t-test" if not equal_var else "Student's t-test",
            statistic=t_stat,
            p_value=p_value,
            effect_size=cohens_d,
            confidence_interval=ci,
            is_significant=p_value < self.alpha
        )
    
    def mann_whitney_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray
    ) -> StatisticalTestResult:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test).
        
        Args:
            group1: First group's samples
            group2: Second group's samples
            
        Returns:
            StatisticalTestResult object
        """
        u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        
        # Effect size (r = Z / sqrt(N))
        n1, n2 = len(group1), len(group2)
        z_score = stats.norm.ppf(1 - p_value/2) if p_value < 1 else 0
        effect_size = z_score / np.sqrt(n1 + n2)
        
        # Bootstrap confidence interval for median difference
        ci = self._bootstrap_ci(group1, group2, np.median)
        
        return StatisticalTestResult(
            test_name="Mann-Whitney U test",
            statistic=u_stat,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            is_significant=p_value < self.alpha
        )
    
    def _bootstrap_ci(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        stat_func: Callable,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for difference in statistics.
        
        Args:
            group1: First group's samples
            group2: Second group's samples
            stat_func: Statistic function (e.g., np.mean, np.median)
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Tuple of (lower, upper) confidence interval bounds
        """
        np.random.seed(42)
        differences = []
        
        for _ in range(n_bootstrap):
            sample1 = np.random.choice(group1, size=len(group1), replace=True)
            sample2 = np.random.choice(group2, size=len(group2), replace=True)
            differences.append(stat_func(sample1) - stat_func(sample2))
        
        lower = np.percentile(differences, 100 * self.alpha / 2)
        upper = np.percentile(differences, 100 * (1 - self.alpha / 2))
        
        return (lower, upper)
    
    def compare_agents(
        self,
        agent_rewards: Dict[str, np.ndarray],
        baseline_name: str = "random"
    ) -> Dict[str, StatisticalTestResult]:
        """
        Compare multiple agents against a baseline using statistical tests.
        
        Args:
            agent_rewards: Dictionary mapping agent names to reward arrays
            baseline_name: Name of the baseline agent
            
        Returns:
            Dictionary of test results for each comparison
        """
        if baseline_name not in agent_rewards:
            raise ValueError(f"Baseline '{baseline_name}' not found in agent_rewards")
        
        baseline = np.array(agent_rewards[baseline_name])
        results = {}
        
        for name, rewards in agent_rewards.items():
            if name == baseline_name:
                continue
            
            rewards = np.array(rewards)
            
            # Perform t-test
            t_result = self.independent_t_test(rewards, baseline)
            
            # Also perform non-parametric test
            mw_result = self.mann_whitney_test(rewards, baseline)
            
            results[f"{name}_vs_{baseline_name}_ttest"] = t_result
            results[f"{name}_vs_{baseline_name}_mannwhitney"] = mw_result
        
        return results
    
    def print_comparison_report(
        self,
        results: Dict[str, StatisticalTestResult]
    ) -> str:
        """
        Print formatted comparison report.
        
        Args:
            results: Dictionary of test results
            
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 70,
            "STATISTICAL SIGNIFICANCE ANALYSIS",
            "=" * 70,
            f"Significance level (α): {self.alpha}",
            ""
        ]
        
        for name, result in results.items():
            lines.append(f"\n{name}:")
            lines.append(f"  Test: {result.test_name}")
            lines.append(f"  Test Statistic: {result.statistic:.4f}")
            lines.append(f"  P-value: {result.p_value:.6f}")
            lines.append(f"  Effect Size: {result.effect_size:.4f}")
            lines.append(f"  95% CI: ({result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f})")
            lines.append(f"  Significant: {'Yes ✓' if result.is_significant else 'No ✗'}")
        
        lines.extend(["", "=" * 70])
        
        report = "\n".join(lines)
        print(report)
        return report


class HyperparameterAnalyzer:
    """
    Analyzes sensitivity of training to hyperparameter choices.
    """
    
    def __init__(self, output_dir: str = "figures"):
        """
        Initialize analyzer.
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}
    
    def run_sensitivity_study(
        self,
        trainer_class,
        param_name: str,
        param_values: List,
        num_episodes: int = 200,
        num_seeds: int = 3,
        **trainer_kwargs
    ) -> Dict:
        """
        Run hyperparameter sensitivity study.
        
        Args:
            trainer_class: Trainer class to instantiate
            param_name: Name of hyperparameter to vary
            param_values: List of values to test
            num_episodes: Episodes per run
            num_seeds: Random seeds per configuration
            **trainer_kwargs: Additional trainer arguments
            
        Returns:
            Dictionary of results
        """
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER SENSITIVITY STUDY: {param_name}")
        print(f"{'='*60}")
        print(f"Testing values: {param_values}")
        print(f"Episodes per run: {num_episodes}")
        print(f"Seeds per config: {num_seeds}")
        
        results = {
            'param_name': param_name,
            'param_values': param_values,
            'rewards': {},
            'final_rewards': {},
            'success_rates': {}
        }
        
        for value in param_values:
            print(f"\n--- Testing {param_name}={value} ---")
            
            value_rewards = []
            value_final_rewards = []
            value_success_rates = []
            
            for seed in range(num_seeds):
                # Modify config with parameter value
                from config import get_config
                config = get_config()
                
                # Set the hyperparameter
                if hasattr(config.agent, param_name):
                    setattr(config.agent, param_name, value)
                elif hasattr(config.env, param_name):
                    setattr(config.env, param_name, value)
                
                config.seed = 42 + seed
                
                # Create trainer
                trainer = trainer_class(config=config, **trainer_kwargs)
                
                # Train
                metrics = trainer.train(num_episodes=num_episodes, verbose=False)
                
                # Collect results
                rewards = metrics.episode_rewards
                value_rewards.extend(rewards)
                value_final_rewards.append(np.mean(rewards[-50:]))
                
                # Evaluate success rate
                eval_results = trainer.evaluate(num_episodes=20)
                value_success_rates.append(eval_results['success_rate'])
                
                print(f"  Seed {seed}: Final avg reward = {np.mean(rewards[-50:]):.2f}")
            
            results['rewards'][value] = value_rewards
            results['final_rewards'][value] = value_final_rewards
            results['success_rates'][value] = value_success_rates
        
        self.results[param_name] = results
        return results
    
    def plot_sensitivity(
        self,
        param_name: str,
        save: bool = True
    ) -> plt.Figure:
        """
        Plot sensitivity analysis results.
        
        Args:
            param_name: Hyperparameter to plot
            save: Whether to save figure
            
        Returns:
            Matplotlib figure
        """
        if param_name not in self.results:
            raise ValueError(f"No results for {param_name}. Run sensitivity study first.")
        
        results = self.results[param_name]
        param_values = results['param_values']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Final rewards boxplot
        final_rewards_data = [results['final_rewards'][v] for v in param_values]
        bp = axes[0].boxplot(final_rewards_data, labels=[str(v) for v in param_values], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.7)
        axes[0].set_xlabel(param_name, fontsize=12)
        axes[0].set_ylabel('Final Avg Reward (last 50 ep)', fontsize=12)
        axes[0].set_title('Reward Sensitivity', fontsize=14, fontweight='bold')
        
        # Plot 2: Success rate
        success_means = [np.mean(results['success_rates'][v]) for v in param_values]
        success_stds = [np.std(results['success_rates'][v]) for v in param_values]
        x = range(len(param_values))
        axes[1].bar(x, success_means, yerr=success_stds, capsize=5, color='green', alpha=0.7)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([str(v) for v in param_values])
        axes[1].set_xlabel(param_name, fontsize=12)
        axes[1].set_ylabel('Success Rate', fontsize=12)
        axes[1].set_title('Success Rate Sensitivity', fontsize=14, fontweight='bold')
        axes[1].set_ylim(0, 1.1)
        
        # Plot 3: Learning curves comparison
        for v in param_values:
            rewards = results['rewards'][v]
            # Average across seeds
            rewards_per_seed = np.array_split(rewards, len(results['final_rewards'][v]))
            min_len = min(len(r) for r in rewards_per_seed)
            rewards_aligned = np.array([r[:min_len] for r in rewards_per_seed])
            mean_curve = np.mean(rewards_aligned, axis=0)
            window = min(20, len(mean_curve) // 5)
            if window > 0:
                smoothed = np.convolve(mean_curve, np.ones(window)/window, mode='valid')
                axes[2].plot(smoothed, label=f'{param_name}={v}', linewidth=2)
        
        axes[2].set_xlabel('Episode', fontsize=12)
        axes[2].set_ylabel('Average Reward', fontsize=12)
        axes[2].set_title('Learning Curves', fontsize=14, fontweight='bold')
        axes[2].legend()
        
        plt.tight_layout()
        
        if save:
            fig.savefig(os.path.join(self.output_dir, f'sensitivity_{param_name}.png'), dpi=150)
        
        return fig
    
    def plot_all_sensitivities(self) -> None:
        """Plot all sensitivity analyses."""
        for param_name in self.results:
            self.plot_sensitivity(param_name)
            print(f"  ✓ Sensitivity plot for {param_name}")


class TrainingVisualizer:
    """Visualizes training metrics and results."""
    
    def __init__(self, log_dir: str = "logs", output_dir: str = "figures"):
        """
        Initialize visualizer.
        
        Args:
            log_dir: Directory containing training logs
            output_dir: Directory to save figures
        """
        self.log_dir = log_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.metrics = None
        self.stat_analyzer = StatisticalAnalyzer()
    
    def load_metrics(self, metrics_file: str = "training_metrics.json") -> Dict:
        """Load training metrics from file."""
        path = os.path.join(self.log_dir, metrics_file)
        with open(path, 'r') as f:
            self.metrics = json.load(f)
        return self.metrics
    
    def plot_learning_curve(
        self,
        window: int = 50,
        save: bool = True,
        show_confidence: bool = True
    ) -> plt.Figure:
        """
        Plot learning curve with optional confidence bands.
        
        Args:
            window: Moving average window size
            save: Whether to save figure
            show_confidence: Whether to show confidence intervals
            
        Returns:
            Matplotlib figure
        """
        if self.metrics is None:
            self.load_metrics()
        
        rewards = np.array(self.metrics['episode_rewards'])
        episodes = np.arange(len(rewards))
        
        # Calculate moving average
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot raw rewards
        ax.plot(episodes, rewards, alpha=0.2, color='steelblue', label='Episode Reward')
        
        # Plot moving average
        ax.plot(episodes[window-1:], moving_avg, color='darkblue', 
                linewidth=2, label=f'{window}-Episode Moving Avg')
        
        # Add confidence bands
        if show_confidence and len(rewards) > window * 2:
            # Calculate rolling std
            rolling_std = np.array([np.std(rewards[max(0,i-window):i+1]) 
                                   for i in range(window-1, len(rewards))])
            ax.fill_between(episodes[window-1:], 
                           moving_avg - 1.96 * rolling_std / np.sqrt(window),
                           moving_avg + 1.96 * rolling_std / np.sqrt(window),
                           alpha=0.3, color='steelblue', label='95% CI')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title('Learning Curve: Episode Rewards', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(os.path.join(self.output_dir, 'learning_curve.png'), dpi=150)
        
        return fig
    
    def plot_epsilon_decay(self, save: bool = True) -> plt.Figure:
        """Plot epsilon decay over training."""
        if self.metrics is None:
            self.load_metrics()
        
        epsilons = np.array(self.metrics['epsilons'])
        episodes = np.arange(len(epsilons))
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(episodes, epsilons, color='green', linewidth=2)
        ax.fill_between(episodes, 0, epsilons, alpha=0.2, color='green')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Epsilon (Exploration Rate)', fontsize=12)
        ax.set_title('Exploration Rate Decay', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(os.path.join(self.output_dir, 'epsilon_decay.png'), dpi=150)
        
        return fig
    
    def plot_loss_curve(self, window: int = 50, save: bool = True) -> plt.Figure:
        """Plot training loss over time."""
        if self.metrics is None:
            self.load_metrics()
        
        losses = np.array(self.metrics['episode_losses'])
        losses = losses[losses > 0]  # Remove zeros
        
        if len(losses) == 0:
            print("No loss data available")
            return None
        
        episodes = np.arange(len(losses))
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(episodes, losses, alpha=0.3, color='red', label='Episode Loss')
        ax.plot(episodes[window-1:], moving_avg, color='darkred',
                linewidth=2, label=f'{window}-Episode Moving Avg')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Average Loss', fontsize=12)
        ax.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(os.path.join(self.output_dir, 'loss_curve.png'), dpi=150)
        
        return fig
    
    def plot_performance_metrics(self, window: int = 50, save: bool = True) -> plt.Figure:
        """Plot multiple performance metrics."""
        if self.metrics is None:
            self.load_metrics()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Attacks contained
        contained = np.array(self.metrics['attacks_contained'])
        moving_avg = np.convolve(contained, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(contained, alpha=0.3, color='green')
        axes[0, 0].plot(np.arange(window-1, len(contained)), moving_avg, 
                       color='darkgreen', linewidth=2)
        axes[0, 0].set_title('Attacks Contained per Episode', fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        
        # False positives
        fps = np.array(self.metrics['false_positives'])
        moving_avg = np.convolve(fps, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(fps, alpha=0.3, color='orange')
        axes[0, 1].plot(np.arange(window-1, len(fps)), moving_avg,
                       color='darkorange', linewidth=2)
        axes[0, 1].set_title('False Positives per Episode', fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        
        # Episode lengths
        lengths = np.array(self.metrics['episode_lengths'])
        moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(lengths, alpha=0.3, color='purple')
        axes[1, 0].plot(np.arange(window-1, len(lengths)), moving_avg,
                       color='darkviolet', linewidth=2)
        axes[1, 0].set_title('Episode Length', fontweight='bold')
        axes[1, 0].set_xlabel('Episode')
        
        # Data loss events
        data_loss = np.array(self.metrics.get('data_loss_events', [0]*len(contained)))
        moving_avg = np.convolve(data_loss, np.ones(window)/window, mode='valid')
        axes[1, 1].plot(data_loss, alpha=0.3, color='red')
        axes[1, 1].plot(np.arange(window-1, len(data_loss)), moving_avg,
                       color='darkred', linewidth=2)
        axes[1, 1].set_title('Data Loss Events per Episode', fontweight='bold')
        axes[1, 1].set_xlabel('Episode')
        
        plt.tight_layout()
        
        if save:
            fig.savefig(os.path.join(self.output_dir, 'performance_metrics.png'), dpi=150)
        
        return fig
    
    def plot_reward_distribution(self, save: bool = True) -> plt.Figure:
        """Plot distribution of episode rewards with statistical test."""
        if self.metrics is None:
            self.load_metrics()
        
        rewards = np.array(self.metrics['episode_rewards'])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(rewards, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
        axes[0].axvline(np.mean(rewards), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(rewards):.1f}')
        axes[0].axvline(np.median(rewards), color='green', linestyle='--',
                       label=f'Median: {np.median(rewards):.1f}')
        axes[0].set_xlabel('Episode Reward', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Reward Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        
        # First half vs second half with statistical test
        half = len(rewards) // 2
        first_half = rewards[:half]
        second_half = rewards[half:]
        
        # Perform t-test
        test_result = self.stat_analyzer.independent_t_test(second_half, first_half)
        
        data = [first_half, second_half]
        labels = ['First Half', 'Second Half']
        
        bp = axes[1].boxplot(data, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # Add significance annotation
        sig_text = f"p={test_result.p_value:.4f}"
        if test_result.is_significant:
            sig_text += " *"
        axes[1].annotate(sig_text, xy=(1.5, max(np.max(first_half), np.max(second_half))),
                        ha='center', fontsize=11)
        
        axes[1].set_ylabel('Episode Reward', fontsize=12)
        axes[1].set_title('Reward Comparison: First vs Second Half', 
                         fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            fig.savefig(os.path.join(self.output_dir, 'reward_distribution.png'), dpi=150)
        
        return fig
    
    def create_summary_report(self, save: bool = True) -> str:
        """Create text summary of training results with statistics."""
        if self.metrics is None:
            self.load_metrics()
        
        rewards = np.array(self.metrics['episode_rewards'])
        
        # Perform statistical test between first and second half
        half = len(rewards) // 2
        first_half = rewards[:half]
        second_half = rewards[half:]
        test_result = self.stat_analyzer.independent_t_test(second_half, first_half)
        
        # Calculate statistics
        stats_dict = {
            'Total Episodes': len(rewards),
            'Final Avg Reward (last 100)': np.mean(rewards[-100:]),
            'Best Episode Reward': np.max(rewards),
            'Worst Episode Reward': np.min(rewards),
            'Overall Mean Reward': np.mean(rewards),
            'Overall Std Reward': np.std(rewards),
            'Total Attacks Contained': sum(self.metrics['attacks_contained']),
            'Total False Positives': sum(self.metrics['false_positives']),
            'Final Epsilon': self.metrics['epsilons'][-1] if self.metrics['epsilons'] else 0
        }
        
        # Format report
        lines = [
            "=" * 60,
            "TRAINING SUMMARY REPORT",
            "=" * 60,
            ""
        ]
        
        for key, value in stats_dict.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")
        
        # Add statistical analysis
        lines.extend([
            "",
            "-" * 60,
            "STATISTICAL ANALYSIS (First Half vs Second Half)",
            "-" * 60,
            f"First Half Mean: {np.mean(first_half):.4f}",
            f"Second Half Mean: {np.mean(second_half):.4f}",
            f"Improvement: {np.mean(second_half) - np.mean(first_half):.4f}",
            f"T-Statistic: {test_result.statistic:.4f}",
            f"P-Value: {test_result.p_value:.6f}",
            f"Effect Size (Cohen's d): {test_result.effect_size:.4f}",
            f"95% CI: ({test_result.confidence_interval[0]:.4f}, {test_result.confidence_interval[1]:.4f})",
            f"Statistically Significant: {'Yes ✓' if test_result.is_significant else 'No ✗'}",
            "",
            "=" * 60
        ])
        
        report = "\n".join(lines)
        print(report)
        
        if save:
            with open(os.path.join(self.output_dir, 'training_summary.txt'), 'w') as f:
                f.write(report)
        
        return report
    
    def plot_all(self) -> None:
        """Generate all visualization plots."""
        print("Generating all visualizations...")
        
        self.plot_learning_curve()
        print("  ✓ Learning curve")
        
        self.plot_epsilon_decay()
        print("  ✓ Epsilon decay")
        
        self.plot_loss_curve()
        print("  ✓ Loss curve")
        
        self.plot_performance_metrics()
        print("  ✓ Performance metrics")
        
        self.plot_reward_distribution()
        print("  ✓ Reward distribution")
        
        self.create_summary_report()
        print("  ✓ Summary report (with statistical analysis)")
        
        print(f"\nAll figures saved to: {self.output_dir}/")


class PolicyAnalyzer:
    """Analyzes learned policy behavior."""
    
    def __init__(self, agent, env, output_dir: str = "figures"):
        """
        Initialize policy analyzer.
        
        Args:
            agent: Trained DQN agent
            env: Environment instance
            output_dir: Directory to save figures
        """
        self.agent = agent
        self.env = env
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def analyze_action_preferences(
        self,
        num_samples: int = 1000,
        save: bool = True
    ) -> plt.Figure:
        """Analyze which actions the agent prefers in different states."""
        actions_taken = []
        login_rates = []
        file_rates = []
        
        for _ in range(num_samples):
            state = self.env.observation_space.sample()
            action = self.agent.select_action(state, training=False)
            
            actions_taken.append(action)
            login_rates.append(state[0])
            file_rates.append(state[1])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Action distribution
        action_counts = np.bincount(actions_taken, minlength=5)
        action_names = ['Do Nothing', 'Block IP', 'Lock Account', 
                       'Terminate Process', 'Isolate Host']
        
        axes[0].bar(action_names, action_counts, color=sns.color_palette("husl", 5))
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Action Distribution', fontsize=14, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Action by threat level
        actions_np = np.array(actions_taken)
        login_np = np.array(login_rates)
        
        low_threat = actions_np[login_np < 30]
        high_threat = actions_np[login_np >= 30]
        
        x = np.arange(5)
        width = 0.35
        
        low_counts = np.bincount(low_threat, minlength=5) if len(low_threat) > 0 else np.zeros(5)
        high_counts = np.bincount(high_threat, minlength=5) if len(high_threat) > 0 else np.zeros(5)
        
        axes[1].bar(x - width/2, low_counts, width, label='Low Threat', color='green', alpha=0.7)
        axes[1].bar(x + width/2, high_counts, width, label='High Threat', color='red', alpha=0.7)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(action_names, rotation=45)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Actions by Threat Level', fontsize=14, fontweight='bold')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save:
            fig.savefig(os.path.join(self.output_dir, 'action_preferences.png'), dpi=150)
        
        return fig
    
    def plot_q_value_heatmap(
        self,
        login_range: Tuple[float, float] = (0, 100),
        file_range: Tuple[float, float] = (0, 200),
        resolution: int = 20,
        save: bool = True
    ) -> plt.Figure:
        """Plot Q-value heatmap for state space."""
        login_vals = np.linspace(login_range[0], login_range[1], resolution)
        file_vals = np.linspace(file_range[0], file_range[1], resolution)
        
        max_q_values = np.zeros((resolution, resolution))
        best_actions = np.zeros((resolution, resolution))
        
        # Get state size from environment
        state_size = self.env.observation_space.shape[0]
        
        for i, login in enumerate(login_vals):
            for j, file_rate in enumerate(file_vals):
                # Create appropriately sized state
                if state_size == 10:
                    state = np.array([login, file_rate, 50.0, 0.0, 0.0, 0.0, login, file_rate, 0.0, 0.5], dtype=np.float32)
                else:
                    state = np.array([login, file_rate, 50.0, 0.5], dtype=np.float32)
                    
                q_values = self.agent.get_q_values(state)
                max_q_values[i, j] = np.max(q_values)
                best_actions[i, j] = np.argmax(q_values)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Max Q-value heatmap
        im1 = axes[0].imshow(max_q_values.T, origin='lower', aspect='auto',
                            extent=[login_range[0], login_range[1], 
                                   file_range[0], file_range[1]],
                            cmap='viridis')
        axes[0].set_xlabel('Login Attempts', fontsize=12)
        axes[0].set_ylabel('File Access Rate', fontsize=12)
        axes[0].set_title('Max Q-Value Landscape', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=axes[0], label='Q-Value')
        
        # Best action heatmap
        im2 = axes[1].imshow(best_actions.T, origin='lower', aspect='auto',
                            extent=[login_range[0], login_range[1],
                                   file_range[0], file_range[1]],
                            cmap='Set3')
        axes[1].set_xlabel('Login Attempts', fontsize=12)
        axes[1].set_ylabel('File Access Rate', fontsize=12)
        axes[1].set_title('Optimal Action Map', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(im2, ax=axes[1], ticks=[0, 1, 2, 3, 4])
        cbar.set_ticklabels(['Nothing', 'Block IP', 'Lock Acct', 'Kill Proc', 'Isolate'])
        
        plt.tight_layout()
        
        if save:
            fig.savefig(os.path.join(self.output_dir, 'q_value_heatmap.png'), dpi=150)
        
        return fig


def plot_baseline_comparison(
    comparison_results: Dict,
    output_path: str = "figures/baseline_comparison.png",
    include_significance: bool = True
) -> plt.Figure:
    """
    Plot comparison of agent with baselines, including significance testing.
    
    Args:
        comparison_results: Dictionary with rewards for each agent
        output_path: Path to save figure
        include_significance: Whether to include statistical significance
        
    Returns:
        Matplotlib figure
    """
    agents = list(comparison_results.keys())
    
    # Check if we have full reward arrays or just summary stats
    if isinstance(comparison_results[agents[0]], dict):
        rewards = [comparison_results[a].get('avg_reward', 0) for a in agents]
        stds = [comparison_results[a].get('std_reward', 0) for a in agents]
    else:
        rewards = [np.mean(comparison_results[a]) for a in agents]
        stds = [np.std(comparison_results[a]) for a in agents]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['steelblue', 'gray', 'orange', 'red', 'green']
    bars = ax.bar(agents, rewards, yerr=stds, capsize=5, 
                 color=colors[:len(agents)], alpha=0.8)
    
    # Add value labels on bars
    for bar, reward in zip(bars, rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{reward:.1f}',
               ha='center', va='bottom', fontsize=12)
    
    # Add significance markers if requested
    if include_significance and 'dqn' in comparison_results and 'random' in comparison_results:
        analyzer = StatisticalAnalyzer()
        
        dqn_rewards = comparison_results['dqn']
        if isinstance(dqn_rewards, dict):
            dqn_rewards = np.random.normal(dqn_rewards['avg_reward'], dqn_rewards.get('std_reward', 1), 100)
        
        random_rewards = comparison_results['random']
        if isinstance(random_rewards, dict):
            random_rewards = np.random.normal(random_rewards['avg_reward'], random_rewards.get('std_reward', 1), 100)
        
        result = analyzer.independent_t_test(np.array(dqn_rewards), np.array(random_rewards))
        if result.is_significant:
            ax.annotate('*', xy=(0.5, max(rewards) * 1.1), fontsize=20, ha='center')
    
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Agent Performance Comparison', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path, dpi=150)
    
    return fig


if __name__ == "__main__":
    # Test visualization and statistical analysis
    print("Testing Evaluation Module")
    print("=" * 50)
    
    # Create sample metrics
    np.random.seed(42)
    sample_metrics = {
        'episode_rewards': list(np.random.randn(500).cumsum() + np.linspace(0, 50, 500)),
        'episode_lengths': list(np.random.randint(50, 100, 500)),
        'episode_losses': list(np.exp(-np.linspace(0, 3, 500)) + np.random.randn(500) * 0.1),
        'epsilons': list(np.exp(-np.linspace(0, 5, 500))),
        'attacks_contained': list(np.random.binomial(3, 0.3, 500)),
        'false_positives': list(np.random.binomial(2, 0.2, 500)),
        'data_loss_events': list(np.random.binomial(1, 0.1, 500))
    }
    
    # Save sample metrics
    os.makedirs('logs', exist_ok=True)
    with open('logs/training_metrics.json', 'w') as f:
        json.dump(sample_metrics, f)
    
    # Test statistical analyzer
    print("\n--- Testing Statistical Analyzer ---")
    analyzer = StatisticalAnalyzer()
    
    group1 = np.random.normal(50, 10, 100)  # DQN agent
    group2 = np.random.normal(30, 15, 100)  # Random agent
    
    result = analyzer.independent_t_test(group1, group2)
    print(result)
    
    result_mw = analyzer.mann_whitney_test(group1, group2)
    print(result_mw)
    
    # Test visualizations
    print("\n--- Testing Visualizations ---")
    visualizer = TrainingVisualizer()
    visualizer.plot_all()
    
    print("\nEvaluation module test complete!")
