# scripts/train.py
"""
Complete training script for DQN agent on traffic signal control task
Usage: python scripts/train.py --episodes 500 --use-cnn --dueling
"""

import argparse
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

from traffic_env import TrafficIntersection
from dqn_cnn_agent import DoubleDQNAgent
from config import MODELS_DIR, TRAINING_CONFIG, DQN_CONFIG, DEFAULT_CONFIG
import numpy as np
from tqdm import tqdm
import json


def train_agent(episodes: int = 500,
                steps_per_episode: int = 500,
                use_cnn: bool = False,
                dueling: bool = False,
                save_every: int = 50,
                save_dir: str = None):
    """
    Train DQN agent on traffic signal control
    
    Args:
        episodes: Number of training episodes
        steps_per_episode: Steps per episode
        use_cnn: Whether to use CNN-based agent
        dueling: Whether to use dueling architecture
        save_every: Save model every N episodes
        save_dir: Directory to save models
    """
    
    if save_dir is None:
        save_dir = MODELS_DIR
    else:
        save_dir = Path(save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize environment
    env = TrafficIntersection(
        arrival_rates=DEFAULT_CONFIG,
        green_duration=DEFAULT_CONFIG['green_duration']
    )
    
    # Initialize agent
    agent = DoubleDQNAgent(
        num_actions=2,
        use_cnn=use_cnn,
        dueling=dueling,
        **DQN_CONFIG
    )
    
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
    print(f"Steps/Episode: {steps_per_episode}")
    print(f"Use CNN: {use_cnn}")
    print(f"Dueling Architecture: {dueling}")
    print(f"Agent Type: {'Double DQN' if not use_cnn else 'CNN Double DQN'}")
    print(f"Learning Rate: {DQN_CONFIG['learning_rate']}")
    print(f"Discount Factor (γ): {DQN_CONFIG['gamma']}")
    print(f"{'='*60}\n")
    
    # Training metrics
    episode_rewards = []
    episode_queue_lengths = []
    episode_throughputs = []
    all_metrics_history = []
    
    # Training loop
    for episode in tqdm(range(episodes), desc="Training Progress", unit="episode"):
        state = env.reset()
        
        if use_cnn:
            state = env.get_grid_observation()
        
        episode_reward = 0
        episode_metrics = []
        
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
            if agent.steps % agent.update_frequency == 0 and len(agent.memory) >= agent.batch_size:
                agent.replay(batch_size=agent.batch_size)
            
            # Update target network periodically
            if agent.steps % agent.target_update_frequency == 0:
                agent.update_target_network()
            
            episode_reward += reward
            agent.steps += 1
            state = next_state
            
            # Record metrics
            metrics = env.get_metrics()
            metrics['step'] = step
            metrics['action'] = action
            metrics['reward'] = reward
            episode_metrics.append(metrics)
        
        # Episode end
        agent.decay_epsilon()
        
        episode_rewards.append(episode_reward)
        avg_queue = np.mean([m['total_queue_length'] for m in episode_metrics])
        total_throughput = episode_metrics[-1]['total_throughput']
        
        episode_queue_lengths.append(avg_queue)
        episode_throughputs.append(total_throughput)
        all_metrics_history.extend(episode_metrics)
        
        # Log progress
        if (episode + 1) % TRAINING_CONFIG['log_every'] == 0:
            avg_reward = np.mean(episode_rewards[-TRAINING_CONFIG['log_every']:])
            avg_queue_recent = np.mean(episode_queue_lengths[-TRAINING_CONFIG['log_every']:])
            
            print(f"\nEpisode {episode + 1}/{episodes}")
            print(f"  Avg Reward: {avg_reward:8.2f}")
            print(f"  Avg Queue: {avg_queue_recent:8.2f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Total Steps: {agent.steps}")
        
        # Save model
        if (episode + 1) % save_every == 0:
            model_path = save_dir / f'agent_ep{episode + 1}_cnn{use_cnn}_dueling{dueling}.h5'
            agent.save_model(str(model_path))
    
    # Save final model
    final_model_path = save_dir / f'agent_final_cnn{use_cnn}_dueling{dueling}.h5'
    agent.save_model(str(final_model_path))
    
    # Save training metrics
    metrics_dict = {
        'episode_rewards': episode_rewards,
        'episode_queue_lengths': episode_queue_lengths,
        'episode_throughputs': episode_throughputs,
        'final_epsilon': float(agent.epsilon),
        'total_steps': agent.steps,
        'total_updates': agent.updates,
        'config': {
            'use_cnn': use_cnn,
            'dueling': dueling,
            'episodes': episodes,
            'steps_per_episode': steps_per_episode
        }
    }
    
    metrics_path = save_dir / f'training_metrics_cnn{use_cnn}_dueling{dueling}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Final Epsilon: {agent.epsilon:.4f}")
    print(f"Total Steps: {agent.steps}")
    print(f"Total Updates: {agent.updates}")
    print(f"Model saved to: {final_model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"{'='*60}\n")
    
    return agent, metrics_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN agent for traffic signal control')
    parser.add_argument('--episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--steps', type=int, default=500, help='Steps per episode')
    parser.add_argument('--use-cnn', action='store_true', help='Use CNN-based agent')
    parser.add_argument('--dueling', action='store_true', help='Use dueling architecture')
    parser.add_argument('--save-every', type=int, default=50, help='Save model every N episodes')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save models')
    
    args = parser.parse_args()
    
    agent, metrics = train_agent(
        episodes=args.episodes,
        steps_per_episode=args.steps,
        use_cnn=args.use_cnn,
        dueling=args.dueling,
        save_every=args.save_every,
        save_dir=args.save_dir
    )
