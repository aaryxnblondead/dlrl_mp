# backend/config.py
import os
from pathlib import Path

# Environment
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
ENV = os.getenv('ENV', 'development')

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'backend' / 'models'
LOGS_DIR = DATA_DIR / 'training_logs'

# Flask Config
FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))

# RL Training Config
DEFAULT_CONFIG = {
    'north_traffic': 0.3,
    'south_traffic': 0.3,
    'east_traffic': 0.25,
    'west_traffic': 0.25,
    'green_duration': 30,
    'use_rl': True,
    'use_cnn': False
}

# DQN Hyperparameters
DQN_CONFIG = {
    'learning_rate': 0.001,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'memory_size': 2000,
    'batch_size': 32,
    'update_frequency': 4,
    'target_update_frequency': 10
}

# Training Config
TRAINING_CONFIG = {
    'episodes': 500,
    'steps_per_episode': 500,
    'save_every': 50,
    'log_every': 10
}

# Paths
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
