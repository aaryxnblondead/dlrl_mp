#!/usr/bin/env python3
"""
Standalone Training Script for Traffic Signal DQN Agent
Usage: python train_model.py [--episodes 500] [--steps 1000] [--use-cnn]
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from traffic_env import TrafficIntersection
from dqn_agent import DQNAgent

def train_agent(episodes: int = 500, 
                steps_per_episode: int = 1000,
                use_cnn: bool = False,
                save_every: int = 50):
    """
    Train DQN agent on traffic signal control
    
    Args:
        episodes: Number of training episodes
        steps_per_episode: Steps per episode
        use_cnn: Whether to use CNN-based agent
        save_every: Save model every N episodes
    """
    
    # Create save directory
    save_dir = Path(__file__).parent / 'models'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize environment with default arrival rates
    arrival_rates = {
        'north': {'left': 0.15, 'straight': 0.25, 'right': 0.10},
        'south': {'left': 0.15, 'straight': 0.25, 'right': 0.10},
        'east': {'left': 0.15, 'straight': 0.25, 'right': 0.10},
        'west': {'left': 0.15, 'straight': 0.25, 'right': 0.10},
    }
    
    env = TrafficIntersection(
        arrival_rates=arrival_rates,
        min_green=10,
        yellow_duration=4
    )
    
    # Initialize agent
    agent = DQNAgent(
        num_actions=7,  # 0: keep, 1-6: switch to NS-Straight/NS-Left/NS-Right/EW-Straight/EW-Left/EW-Right
        state_size=44,  # Enhanced state observation
        use_cnn=use_cnn,
        learning_rate=0.0003,
        gamma=0.75,  # Prioritize immediate queue clearing
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995
    )
    
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Episodes:           {episodes}")
    print(f"Steps/Episode:      {steps_per_episode}")
    print(f"Use CNN:            {use_cnn}")
    print(f"State Size:         {44 if not use_cnn else '4x3 grid'}")
    print(f"Actions:            7 (keep, NS-Straight, NS-Left, NS-Right, EW-Straight, EW-Left, EW-Right)")
    print(f"Learning Rate:      0.0003")
    print(f"Discount Factor:    0.75 (prioritizes immediate rewards)")
    print(f"Epsilon Decay:      0.9995")
    print(f"Save Directory:     {save_dir}")
    print("="*70 + "\n")
    
    # Training metrics
    episode_rewards = []
    episode_avg_queues = []
    episode_max_queues = []
    episode_throughputs = []
    episode_losses = []
    
    best_avg_queue = float('inf')
    
    print("Starting training...")
    print("-" * 70)
    
    # Training loop
    for episode in range(episodes):
        print(f"\n[Episode {episode + 1}/{episodes}] Starting... (ε={agent.epsilon:.4f})")
        state = env.reset()
        
        if use_cnn:
            state = env.get_grid_observation()
        
        episode_reward = 0
        episode_queue_samples = []
        episode_loss = []
        
        action_names = ['KEEP', 'NS-Straight', 'NS-Left', 'NS-Right', 'EW-Straight', 'EW-Left', 'EW-Right']
        
        for step in range(steps_per_episode):
            # Choose action
            action = agent.act(state, training=True)
            
            # Execute action
            next_state, reward, done = env.step(action)
            
            if use_cnn:
                next_state = env.get_grid_observation()
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train
            if len(agent.memory) >= agent.batch_size:
                loss = agent.replay(batch_size=agent.batch_size)
                if loss > 0:
                    episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
            
            # Record queue metrics
            metrics = env.get_metrics()
            episode_queue_samples.append(metrics['total_queue_length'])
            
            # Verbose progress every 100 steps
            if (step + 1) % 100 == 0:
                current_avg_queue = np.mean(episode_queue_samples[-100:])
                current_avg_reward = episode_reward / (step + 1)
                print(f"  Ep {episode + 1} Step {step + 1:4d}/{steps_per_episode} | "
                      f"Action: {action_names[action]:12s} | "
                      f"Queue: {current_avg_queue:5.1f} | "
                      f"Reward: {reward:7.1f} | "
                      f"Cumul: {episode_reward:8.1f} | "
                      f"Memory: {len(agent.memory):5d}")
            
            if done:
                break
        
        # Episode end
        agent.decay_epsilon()
        
        # Calculate episode statistics
        avg_queue = np.mean(episode_queue_samples)
        max_queue = np.max(episode_queue_samples)
        throughput = env.vehicles_passed
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        
        episode_rewards.append(episode_reward)
        episode_avg_queues.append(avg_queue)
        episode_max_queues.append(max_queue)
        episode_throughputs.append(throughput)
        episode_losses.append(avg_loss)
        
        # Print episode summary with action distribution
        print(f"\n  Episode Summary:")
        print(f"  - Total Reward: {episode_reward:.1f}")
        print(f"  - Avg Queue: {avg_queue:.1f} | Max Queue: {max_queue:.0f}")
        print(f"  - Throughput: {throughput} vehicles | Avg Loss: {avg_loss:.4f}")
        print(f"  - Action Distribution: ", end="")
        for name, count in action_counts.items():
            if count > 0:
                print(f"{name}:{count} ", end="")
        print()
        
        # Update best model
        if avg_queue < best_avg_queue:
            best_avg_queue = avg_queue
            best_model_path = save_dir / f'agent_best_cnn{use_cnn}.keras'
            agent.q_network.save(str(best_model_path))
        
        # Log progress
        if (episode + 1) % 10 == 0:
            recent_rewards = episode_rewards[-10:]
            recent_queues = episode_avg_queues[-10:]
            recent_throughput = episode_throughputs[-10:]
            
            print(f"Episode {episode + 1:4d}/{episodes} | "
                  f"Reward: {np.mean(recent_rewards):8.1f} | "
                  f"Avg Queue: {np.mean(recent_queues):5.1f} | "
                  f"Max Queue: {max_queue:3.0f} | "
                  f"Throughput: {np.mean(recent_throughput):5.0f} | "
                  f"ε: {agent.epsilon:.4f} | "
                  f"Loss: {avg_loss:.4f}")
        
        # Save model periodically
        if (episode + 1) % save_every == 0:
            checkpoint_path = save_dir / f'agent_ep{episode + 1}_cnn{use_cnn}.keras'
            agent.q_network.save(str(checkpoint_path))
            print(f"  → Checkpoint saved: {checkpoint_path.name}")
    
    print("-" * 70)
    print("\nTraining Complete!")
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = save_dir / f'agent_final_{timestamp}_cnn{use_cnn}.keras'
    agent.q_network.save(str(final_model_path))
    
    # Save training metrics in training_results.json format
    final_avg_reward = np.mean(episode_rewards[-50:])
    final_avg_queue = np.mean(episode_avg_queues[-50:])
    
    training_result = {
        "config_id": 1,
        "config": {
            "north_traffic": arrival_rates['north']['straight'],
            "south_traffic": arrival_rates['south']['straight'],
            "east_traffic": arrival_rates['east']['straight'],
            "west_traffic": arrival_rates['west']['straight'],
            "min_green": env.min_green_duration
        },
        "avg_reward": float(final_avg_reward),
        "avg_queue_length": float(final_avg_queue),
        "punishment": float(abs(final_avg_reward)),
        "config": {
            "use_cnn": use_cnn,
            "episodes": episodes,
            "steps_per_episode": steps_per_episode,
            "learning_rate": 0.0003,
            "gamma": 0.75,
            "state_size": 44,
            "num_actions": 7
        }
    }
    
    # Save to data/training_results/training_results.json
    results_dir = Path(__file__).parent / 'data' / 'training_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'training_results.json'
    
    # Load existing results or create new list
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        # Update config_id to be next available
        training_result['config_id'] = max([r['config_id'] for r in results], default=0) + 1
        results.append(training_result)
    else:
        results = [training_result]
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Final Epsilon:       {agent.epsilon:.4f}")
    print(f"Total Training Steps: {agent.steps}")
    print(f"Best Avg Queue:      {best_avg_queue:.2f}")
    print(f"Final Avg Reward:    {final_avg_reward:.2f}")
    print(f"Final Avg Queue:     {final_avg_queue:.2f}")
    print(f"Final Model:         {final_model_path.name}")
    print(f"Best Model:          agent_best_cnn{use_cnn}.keras")
    print(f"Results File:        {results_path}")
    print("="*70 + "\n")
    
    return agent, training_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train DQN agent for traffic signal control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_model.py                           # Train with default settings
  python train_model.py --episodes 1000           # Train for 1000 episodes
  python train_model.py --use-cnn                 # Use CNN-based agent
  python train_model.py --episodes 500 --steps 2000  # Custom episode length
        """
    )
    
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of training episodes (default: 500)')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Steps per episode (default: 1000)')
    parser.add_argument('--use-cnn', action='store_true',
                        help='Use CNN-based agent (default: MLP)')
    parser.add_argument('--save-every', type=int, default=50,
                        help='Save model every N episodes (default: 50)')
    
    args = parser.parse_args()
    
    try:
        agent, metrics = train_agent(
            episodes=args.episodes,
            steps_per_episode=args.steps,
            use_cnn=args.use_cnn,
            save_every=args.save_every
        )
        print("✓ Training completed successfully!")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
