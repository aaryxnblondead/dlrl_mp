# backend/training_manager.py
import itertools
import json
from pathlib import Path
import numpy as np
import threading
import time
from traffic_env import TrafficIntersection
from dqn_cnn_agent import DoubleDQNAgent
from config import DQN_CONFIG, MODELS_DIR

class TrainingManager:
    def __init__(self, parameter_space, episodes_per_config=10, steps_per_episode=500, sample_configs=None):
        self.parameter_space = parameter_space
        self.episodes_per_config = episodes_per_config
        self.steps_per_episode = steps_per_episode
        self.sample_configs = sample_configs  # If set, randomly sample this many configs instead of exhaustive
        self.results = []
        self.status = "idle"
        self.progress = 0
        self.total_configs = 0
        self.lock = threading.Lock()

    def get_status(self):
        with self.lock:
            return {
                "status": self.status,
                "progress": self.progress,
                "total_configs": self.total_configs,
                "results_count": len(self.results)
            }

    def _run_single_configuration(self, config_id, config):
        """Trains and evaluates the agent for a single configuration."""
        try:
            # Handle both simplified (per-direction) and detailed (per-lane) arrival rates
            if 'north_left' in config:
                # Detailed per-lane configuration
                arrival_rates = {
                    'north': {
                        'left': config['north_left'],
                        'straight': config['north_straight'],
                        'right': config['north_right']
                    },
                    'south': {
                        'left': config['south_left'],
                        'straight': config['south_straight'],
                        'right': config['south_right']
                    },
                    'east': {
                        'left': config['east_left'],
                        'straight': config['east_straight'],
                        'right': config['east_right']
                    },
                    'west': {
                        'left': config['west_left'],
                        'straight': config['west_straight'],
                        'right': config['west_right']
                    }
                }
            else:
                # Simplified per-direction configuration (equal rates for all turns)
                arrival_rates = {
                    'north': {'left': config['north_traffic'], 'straight': config['north_traffic'], 'right': config['north_traffic']},
                    'south': {'left': config['south_traffic'], 'straight': config['south_traffic'], 'right': config['south_traffic']},
                    'east': {'left': config['east_traffic'], 'straight': config['east_traffic'], 'right': config['east_traffic']},
                    'west': {'left': config['west_traffic'], 'straight': config['west_traffic'], 'right': config['west_traffic']}
                }
            
            env = TrafficIntersection(
                arrival_rates=arrival_rates,
                min_green=config['min_green']
            )
            
            agent = DoubleDQNAgent(num_actions=2, **DQN_CONFIG)

            total_rewards = []
            avg_queue_lengths = []

            for episode in range(self.episodes_per_config):
                env.reset()
                state = env.get_grid_observation() if agent.use_cnn else env.get_observation()
                total_reward = 0
                episode_metrics_history = []
                
                for step in range(self.steps_per_episode):
                    action = agent.act(state)
                    _, reward, done = env.step(action)
                    
                    metrics = env.get_metrics()
                    episode_metrics_history.append(metrics)

                    next_state = env.get_grid_observation() if agent.use_cnn else env.get_observation()
                    
                    agent.remember(state, action, reward, next_state, done)
                    agent.replay()
                    
                    state = next_state
                    total_reward += reward
                    
                    if done:
                        break
                
                total_rewards.append(total_reward)
                if episode_metrics_history:
                    avg_queue_lengths.append(np.mean([m['total_queue_length'] for m in episode_metrics_history]))
                else:
                    avg_queue_lengths.append(0)

            # After training, evaluate the final performance
            eval_reward = np.mean(total_rewards[-5:])  # Avg reward of last 5 episodes
            eval_queue = np.mean(avg_queue_lengths[-5:])

            result = {
                "config_id": config_id,
                "config": config,
                "avg_reward": float(eval_reward),
                "avg_queue_length": float(eval_queue),
                "punishment": -float(eval_reward) # Assuming reward is negative
            }
            
            return result

        except Exception as e:
            print(f"Error during training for config {config_id}: {e}")
            return {
                "config_id": config_id,
                "config": config,
                "error": str(e)
            }

    def run_training(self):
        """The main method that iterates through parameter combinations."""
        if self.status == "running":
            return

        with self.lock:
            self.status = "running"
            self.results = []
            self.progress = 0

        # Generate parameter combinations
        keys, values = zip(*self.parameter_space.items())
        
        if self.sample_configs:
            # Random sampling for large spaces
            import random
            param_combinations = []
            for _ in range(self.sample_configs):
                config = {key: random.choice(val) for key, val in self.parameter_space.items()}
                param_combinations.append(config)
        else:
            # Exhaustive grid search for smaller spaces
            param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        with self.lock:
            self.total_configs = len(param_combinations)

        for i, config in enumerate(param_combinations):
            result = self._run_single_configuration(i + 1, config)
            
            with self.lock:
                self.results.append(result)
                self.progress = i + 1
            
            # Save results incrementally
            self.save_results()
            
            # Small delay
            time.sleep(0.1)

        with self.lock:
            self.status = "completed"
        
        # Save final model
        # This part can be improved to save the best model
        if self.results:
            best_config_result = max(self.results, key=lambda x: x.get('avg_reward', -np.inf))
            # Here you would typically re-train with the best config and save that model
            print(f"Best configuration found: {best_config_result['config']}")

    def save_results(self):
        """Saves the collected results to a JSON file."""
        results_dir = Path('data/training_results')
        results_dir.mkdir(exist_ok=True)
        
        with self.lock:
            with open(results_dir / 'training_results.json', 'w') as f:
                json.dump(self.results, f, indent=4)

    def load_results(self):
        """Loads results from a JSON file."""
        results_file = Path('data/training_results/training_results.json')
        if results_file.exists():
            with self.lock:
                with open(results_file, 'r') as f:
                    self.results = json.load(f)
            return self.results
        return []

def start_training_in_background(parameter_space, episodes, steps, sample_configs=None):
    manager = TrainingManager(parameter_space, episodes, steps, sample_configs)
    thread = threading.Thread(target=manager.run_training)
    thread.daemon = True
    thread.start()
    return manager
