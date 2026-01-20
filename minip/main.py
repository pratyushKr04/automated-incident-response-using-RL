"""
Main Entry Point for Automated Incident Response using Reinforcement Learning.

Simplified interface with two commands:
- python main.py preprocess    # Extract features from datasets
- python main.py train         # Train the RL agent (auto-saves figures)
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import get_config
from src.preprocess import preprocess_data
from src.train import Trainer
from src.evaluate import TrainingVisualizer, PolicyAnalyzer


def run_preprocess(args, silent: bool = False):
    """Run data preprocessing pipeline."""
    if not silent:
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
    
    params = preprocess_data()  # Uses default path (project_root/extracted_params.json)
    
    if not silent:
        print("\nPreprocessing complete!")
    return params


def run_train(args):
    """Train the RL agent with automatic visualization."""
    # ALWAYS run preprocessing first
    print("\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    print("Extracting parameters from datasets...")
    
    try:
        run_preprocess(args, silent=False)
    except Exception as e:
        print(f"Warning: Preprocessing issue: {e}")
        print("Continuing with available data...")
    
    print("\n" + "="*60)
    print("STEP 2: TRAINING RL AGENT")
    print("="*60)
    
    trainer = Trainer(
        attack_type=args.attack_type,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        use_prioritized_replay=getattr(args, 'prioritized_replay', False),
        use_n_step=getattr(args, 'n_step', False),
        n_steps=getattr(args, 'n_steps', 3)
    )
    
    metrics = trainer.train(
        num_episodes=args.episodes,
        eval_frequency=args.eval_freq,
        save_frequency=args.save_freq,
        verbose=True
    )
    
    # Final evaluation
    print("\n" + "="*60)
    print("STEP 3: FINAL EVALUATION")
    print("="*60)
    eval_results = trainer.evaluate(num_episodes=100)
    
    print(f"Average Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Success Rate: {eval_results['success_rate']*100:.1f}%")
    print(f"Avg False Positives: {eval_results['avg_false_positives']:.2f}")
    print(f"Avg Attacks Contained: {eval_results['avg_contained']:.2f}")
    
    # Compare with baselines
    if args.compare_baselines:
        print("\n" + "="*60)
        print("STEP 4: BASELINE COMPARISON")
        print("="*60)
        trainer.compare_with_baselines(num_episodes=100)
    
    # Auto-generate visualizations
    print("\n" + "="*60)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("="*60)
    
    try:
        visualizer = TrainingVisualizer(
            log_dir=args.log_dir,
            output_dir=args.output_dir
        )
        visualizer.plot_all()
        print(f"All figures saved to: {args.output_dir}/")
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")
    
    # Policy analysis
    try:
        analyzer = PolicyAnalyzer(trainer.agent, trainer.env, output_dir=args.output_dir)
        analyzer.analyze_action_preferences()
        analyzer.plot_q_value_heatmap()
        print("Policy analysis saved.")
    except Exception as e:
        print(f"Warning: Could not analyze policy: {e}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nOutputs saved to:")
    print(f"  Models: {args.checkpoint_dir}/")
    print(f"  Logs: {args.log_dir}/")
    print(f"  Figures: {args.output_dir}/")
    
    return trainer


def main():
    parser = argparse.ArgumentParser(
        description="Automated Incident Response using Reinforcement Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py preprocess             # Extract features from datasets
  python main.py train --episodes 1000  # Train for 1000 episodes
  python main.py train --compare-baselines  # Train and compare with baselines
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess datasets')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train RL agent')
    train_parser.add_argument('--episodes', type=int, default=500, 
                             help='Number of training episodes')
    train_parser.add_argument('--attack-type', type=str, default='random',
                             choices=['bruteforce', 'ransomware', 'both', 'random'],
                             help='Attack type to train on')
    train_parser.add_argument('--checkpoint-dir', type=str, default='models',
                             help='Directory for model checkpoints')
    train_parser.add_argument('--log-dir', type=str, default='logs',
                             help='Directory for training logs')
    train_parser.add_argument('--output-dir', type=str, default='figures',
                             help='Directory for visualizations')
    train_parser.add_argument('--eval-freq', type=int, default=50,
                             help='Evaluation frequency (episodes)')
    train_parser.add_argument('--save-freq', type=int, default=100,
                             help='Checkpoint save frequency')
    train_parser.add_argument('--prioritized-replay', action='store_true',
                             help='Use prioritized experience replay')
    train_parser.add_argument('--n-step', action='store_true',
                             help='Use N-step returns for better credit assignment')
    train_parser.add_argument('--n-steps', type=int, default=3,
                             help='Number of steps for N-step returns')
    train_parser.add_argument('--compare-baselines', action='store_true',
                             help='Compare with baseline agents after training')
    
    args = parser.parse_args()
    
    # Default to 'train' if no command specified
    if args.command is None:
        parser.print_help()
        return
    
    # Run appropriate command
    commands = {
        'preprocess': run_preprocess,
        'train': run_train,
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
