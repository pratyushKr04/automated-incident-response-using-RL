"""
Main Entry Point for Automated Incident Response using Reinforcement Learning.

This script provides a unified interface to:
- Preprocess datasets
- Train the RL agent
- Evaluate trained models
- Generate visualizations
- Run demo simulations

Usage:
    python main.py preprocess    # Extract features from datasets
    python main.py train         # Train the RL agent
    python main.py evaluate      # Evaluate trained model
    python main.py visualize     # Generate training plots
    python main.py demo          # Run interactive demo
    python main.py all           # Run complete pipeline
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


def run_preprocess(args):
    """Run data preprocessing pipeline."""
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    
    params = preprocess_data(save_path="src/extracted_params.json")
    
    print("\nPreprocessing complete!")
    return params


def run_train(args):
    """Train the RL agent."""
    print("\n" + "="*60)
    print("TRAINING RL AGENT")
    print("="*60)
    
    trainer = Trainer(
        attack_type=args.attack_type,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        use_prioritized_replay=args.prioritized_replay
    )
    
    metrics = trainer.train(
        num_episodes=args.episodes,
        eval_frequency=args.eval_freq,
        save_frequency=args.save_freq,
        verbose=True
    )
    
    # Final evaluation
    print("\nRunning final evaluation...")
    eval_results = trainer.evaluate(num_episodes=100)
    
    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS")
    print("="*60)
    print(f"Average Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Success Rate: {eval_results['success_rate']*100:.1f}%")
    print(f"Avg False Positives: {eval_results['avg_false_positives']:.2f}")
    print(f"Avg Attacks Contained: {eval_results['avg_contained']:.2f}")
    
    # Compare with baselines if requested
    if args.compare_baselines:
        print("\nComparing with baseline agents...")
        trainer.compare_with_baselines(num_episodes=100)
    
    return trainer


def run_evaluate(args):
    """Evaluate a trained model."""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    from src.incident_env import IncidentResponseEnv
    from src.agent import DQNAgent
    
    config = get_config()
    
    # Load environment
    env = IncidentResponseEnv(config=config, attack_type=args.attack_type)
    
    # Load agent
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        config=config.agent
    )
    
    model_path = os.path.join(args.checkpoint_dir, args.model_name)
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print(f"Model not found at {model_path}")
        return None
    
    # Run evaluation
    print(f"\nEvaluating model: {model_path}")
    
    rewards = []
    successes = 0
    
    for ep in range(args.num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        
        while True:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        rewards.append(episode_reward)
        stats = info.get('episode_stats', {})
        if stats.get('attacks_contained', 0) > 0 or stats.get('false_positives', 0) == 0:
            successes += 1
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{args.num_episodes}: Reward = {episode_reward:.1f}")
    
    # Print results
    print("\n" + "-"*40)
    print("EVALUATION RESULTS")
    print("-"*40)
    print(f"Episodes: {args.num_episodes}")
    print(f"Average Reward: {sum(rewards)/len(rewards):.2f}")
    print(f"Max Reward: {max(rewards):.2f}")
    print(f"Min Reward: {min(rewards):.2f}")
    print(f"Success Rate: {successes/args.num_episodes*100:.1f}%")
    
    # Policy analysis
    if args.analyze_policy:
        print("\nAnalyzing learned policy...")
        analyzer = PolicyAnalyzer(agent, env, output_dir=args.output_dir)
        analyzer.analyze_action_preferences()
        analyzer.plot_q_value_heatmap()
        print(f"Policy analysis saved to: {args.output_dir}/")
    
    return rewards


def run_visualize(args):
    """Generate training visualizations."""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    visualizer = TrainingVisualizer(
        log_dir=args.log_dir,
        output_dir=args.output_dir
    )
    
    try:
        visualizer.plot_all()
        print(f"\nAll figures saved to: {args.output_dir}/")
    except FileNotFoundError:
        print("Error: Training metrics not found. Run training first.")
        return None
    
    return visualizer


def run_demo(args):
    """Run interactive demonstration."""
    print("\n" + "="*60)
    print("INTERACTIVE DEMO")
    print("="*60)
    
    from src.incident_env import IncidentResponseEnv
    from src.agent import DQNAgent
    
    config = get_config()
    
    # Create environment with rendering
    env = IncidentResponseEnv(
        config=config, 
        attack_type=args.attack_type,
        render_mode="human"
    )
    
    # Try to load trained agent
    agent = None
    model_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    
    if os.path.exists(model_path):
        agent = DQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            config=config.agent
        )
        agent.load(model_path)
        print(f"Loaded trained agent from {model_path}")
    else:
        print("No trained model found. Using random agent.")
    
    # Run demo episodes
    action_names = env.action_names
    
    for episode in range(args.num_episodes):
        print(f"\n{'='*60}")
        print(f"EPISODE {episode + 1}")
        print("="*60)
        
        state, info = env.reset()
        total_reward = 0.0
        step = 0
        
        while True:
            # Select action
            if agent:
                action = agent.select_action(state, training=False)
                q_values = agent.get_q_values(state)
            else:
                action = env.action_space.sample()
                q_values = None
            
            # Display state
            print(f"\nStep {step + 1}")
            print(f"  Observation: Login={state[0]:.0f}, Files={state[1]:.0f}, "
                  f"CPU={state[2]:.1f}%, Time={state[3]:.2f}")
            
            if q_values is not None:
                print(f"  Q-Values: {dict(zip(action_names, q_values.round(2)))}")
            
            print(f"  Action: {action_names[action]}")
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            
            print(f"  Reward: {reward:.2f}")
            print(f"  Attack Active: {info.get('is_attack_active', 'Unknown')}")
            
            total_reward += reward
            state = next_state
            step += 1
            
            if terminated:
                print("\n*** SYSTEM COMPROMISED - Episode ends ***")
                break
            if truncated:
                print("\n*** Episode complete (max steps reached) ***")
                break
            
            # Pause for readability
            if args.interactive:
                input("Press Enter to continue...")
        
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Steps: {step}")
        
        stats = info.get('episode_stats', {})
        print(f"  Attacks Contained: {stats.get('attacks_contained', 0)}")
        print(f"  False Positives: {stats.get('false_positives', 0)}")
    
    env.close()
    print("\nDemo complete!")


def run_all(args):
    """Run complete pipeline."""
    print("\n" + "="*60)
    print("RUNNING COMPLETE PIPELINE")
    print("="*60)
    
    # Step 1: Preprocess
    print("\n[1/4] Preprocessing data...")
    try:
        run_preprocess(args)
    except Exception as e:
        print(f"Preprocessing skipped: {e}")
    
    # Step 2: Train
    print("\n[2/4] Training agent...")
    args.compare_baselines = True
    trainer = run_train(args)
    
    # Step 3: Visualize
    print("\n[3/4] Generating visualizations...")
    run_visualize(args)
    
    # Step 4: Demo
    print("\n[4/4] Running demo...")
    args.num_episodes = 3
    args.interactive = False
    run_demo(args)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Automated Incident Response using Reinforcement Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py preprocess             # Extract features from datasets
  python main.py train --episodes 1000  # Train for 1000 episodes
  python main.py train --attack-type bruteforce  # Train on brute-force attacks
  python main.py evaluate --model best_model.pt  # Evaluate specific model
  python main.py visualize              # Generate training plots
  python main.py demo --interactive     # Run interactive demo
  python main.py all                    # Run complete pipeline
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
    train_parser.add_argument('--eval-freq', type=int, default=50,
                             help='Evaluation frequency (episodes)')
    train_parser.add_argument('--save-freq', type=int, default=100,
                             help='Checkpoint save frequency')
    train_parser.add_argument('--prioritized-replay', action='store_true',
                             help='Use prioritized experience replay')
    train_parser.add_argument('--compare-baselines', action='store_true',
                             help='Compare with baseline agents after training')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model-name', type=str, default='best_model.pt',
                            help='Model file to evaluate')
    eval_parser.add_argument('--checkpoint-dir', type=str, default='models')
    eval_parser.add_argument('--num-episodes', type=int, default=100,
                            help='Number of evaluation episodes')
    eval_parser.add_argument('--attack-type', type=str, default='random',
                            choices=['bruteforce', 'ransomware', 'both', 'random'])
    eval_parser.add_argument('--analyze-policy', action='store_true',
                            help='Analyze learned policy')
    eval_parser.add_argument('--output-dir', type=str, default='figures')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Generate visualizations')
    viz_parser.add_argument('--log-dir', type=str, default='logs')
    viz_parser.add_argument('--output-dir', type=str, default='figures')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run interactive demo')
    demo_parser.add_argument('--num-episodes', type=int, default=5,
                            help='Number of demo episodes')
    demo_parser.add_argument('--attack-type', type=str, default='random',
                            choices=['bruteforce', 'ransomware', 'both', 'random'])
    demo_parser.add_argument('--checkpoint-dir', type=str, default='models')
    demo_parser.add_argument('--interactive', action='store_true',
                            help='Pause after each step')
    
    # All command
    all_parser = subparsers.add_parser('all', help='Run complete pipeline')
    all_parser.add_argument('--episodes', type=int, default=500)
    all_parser.add_argument('--attack-type', type=str, default='random',
                           choices=['bruteforce', 'ransomware', 'both', 'random'])
    all_parser.add_argument('--checkpoint-dir', type=str, default='models')
    all_parser.add_argument('--log-dir', type=str, default='logs')
    all_parser.add_argument('--output-dir', type=str, default='figures')
    all_parser.add_argument('--eval-freq', type=int, default=50)
    all_parser.add_argument('--save-freq', type=int, default=100)
    all_parser.add_argument('--prioritized-replay', action='store_true')
    
    args = parser.parse_args()
    
    # Default to 'all' if no command specified
    if args.command is None:
        parser.print_help()
        return
    
    # Run appropriate command
    commands = {
        'preprocess': run_preprocess,
        'train': run_train,
        'evaluate': run_evaluate,
        'visualize': run_visualize,
        'demo': run_demo,
        'all': run_all
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
