# backend/app.py
"""
Enhanced Flask backend with real-time visualization and metrics
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from traffic_env import TrafficIntersection
from dqn_cnn_agent import DoubleDQNAgent
from dqn_agent import DQNAgent
from visualization import TrainingVisualizer
from config import MODELS_DIR, DEFAULT_CONFIG, DQN_CONFIG
from training_manager import TrainingManager, start_training_in_background
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import threading
import time
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app)

# Global simulation state
simulation_state = {
    'running': False,
    'env': None,
    'agent': None,
    'metrics_history': [],
    'episode': 0,
    'step': 0,
    'use_rl': True,
    'use_cnn': False,
    'use_dueling': False,
    'comparison_fixed_timing': None,
    'agent_metrics': {
        'loss_history': [],
        'q_value_history': [],
        'td_error_history': []
    },
    'start_time': None
}

training_manager = None
visualizer = TrainingVisualizer()


def initialize_simulation(config: dict):
    """Initialize traffic environment and agent"""
    arrival_rates = {
        'north': config.get('north_traffic', 0.3),
        'south': config.get('south_traffic', 0.3),
        'east': config.get('east_traffic', 0.25),
        'west': config.get('west_traffic', 0.25)
    }
    
    simulation_state['env'] = TrafficIntersection(
        arrival_rates=arrival_rates,
        green_duration=config.get('green_duration', 30)
    )
    
    # Initialize agent based on configuration
    use_cnn = config.get('use_cnn', False)
    use_dueling = config.get('use_dueling', False)
    
    agent = DoubleDQNAgent(
        num_actions=2,
        use_cnn=use_cnn,
        dueling=use_dueling,
        **DQN_CONFIG
    )
    
    simulation_state['agent'] = agent
    simulation_state['use_cnn'] = use_cnn
    simulation_state['use_dueling'] = use_dueling
    
    # Try to load pre-trained model
    try:
        model_files = list(MODELS_DIR.glob('agent_final*.h5'))
        if model_files:
            model_path = model_files[0]
            agent.load_model(str(model_path))
            print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Warning: Could not load pre-trained model: {e}")
    
    simulation_state['metrics_history'] = []
    simulation_state['agent_metrics'] = {
        'loss_history': [],
        'q_value_history': [],
        'td_error_history': []
    }
    simulation_state['step'] = 0
    simulation_state['use_rl'] = config.get('use_rl', True)
    simulation_state['start_time'] = datetime.now()


@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize simulation with configuration"""
    try:
        config = request.json
        if config is None:
            config = DEFAULT_CONFIG
        initialize_simulation(config)
        
        return jsonify({
            'status': 'success',
            'message': 'Simulation initialized',
            'config': config
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/api/start', methods=['POST'])
def start_simulation():
    """Start simulation"""
    simulation_state['running'] = True
    simulation_state['episode'] = 0
    
    return jsonify({'status': 'success', 'message': 'Simulation started'})


@app.route('/api/stop', methods=['POST'])
def stop_simulation():
    """Stop simulation"""
    simulation_state['running'] = False
    
    return jsonify({'status': 'success', 'message': 'Simulation stopped'})


@app.route('/api/step', methods=['POST'])
def step_simulation():
    """Execute one simulation step"""
    env = simulation_state['env']
    agent = simulation_state['agent']
    
    if env is None:
        return jsonify({'status': 'error', 'message': 'Simulation not initialized'}), 400
    
    # Get state
    if simulation_state['use_cnn']:
        state = env.get_grid_observation()
    else:
        state = env.get_observation()
    
    # Choose action
    if simulation_state['use_rl']:
        action = agent.act(state, training=False)
    else:
        # Fixed timing: switch at end of green phase
        action = 0 if env.time_in_phase < env.green_duration - 5 else 1
    
    # Execute action
    next_state, reward, done = env.step(action)
    
    # Get metrics
    metrics = env.get_metrics()
    metrics['step'] = simulation_state['step']
    metrics['action'] = action
    metrics['reward'] = reward
    metrics['agent_type'] = 'RL' if simulation_state['use_rl'] else 'Fixed'
    
    simulation_state['metrics_history'].append(metrics)
    simulation_state['step'] += 1
    
    return jsonify(metrics)


@app.route('/api/episode', methods=['POST'])
def run_episode():
    """Run full episode (500 steps by default)"""
    steps = request.json.get('steps', 500) if request.json else 500
    episode_metrics = {
        'steps': [],
        'total_reward': 0,
        'avg_queue': 0,
        'total_throughput': 0
    }
    
    for _ in range(steps):
        response_tuple = step_simulation()
        response = app.make_response(response_tuple)
        if response.status_code == 200:
            metrics = response.get_json()
            episode_metrics['steps'].append(metrics)
            episode_metrics['total_reward'] += metrics.get('reward', 0)
            episode_metrics['total_throughput'] += metrics.get('total_throughput', 0)
    
    # Calculate average queue
    if episode_metrics['steps']:
        episode_metrics['avg_queue'] = np.mean(
            [m['total_queue_length'] for m in episode_metrics['steps']]
        )
    
    simulation_state['episode'] += 1
    
    return jsonify(episode_metrics)


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get current metrics"""
    if not simulation_state['metrics_history']:
        return jsonify({
            'status': 'error',
            'message': 'No metrics available'
        }), 400
    
    latest = simulation_state['metrics_history'][-1]
    
    # Calculate averages over last 50 steps
    recent_metrics = simulation_state['metrics_history'][-50:]
    
    return jsonify({
        'current': latest,
        'averages': {
            'queue_length': np.mean([m['total_queue_length'] for m in recent_metrics]),
            'throughput': np.mean([m['total_throughput'] for m in recent_metrics]),
            'north_queue': np.mean([m['north']['queue_length'] for m in recent_metrics]),
            'south_queue': np.mean([m['south']['queue_length'] for m in recent_metrics]),
            'east_queue': np.mean([m['east']['queue_length'] for m in recent_metrics]),
            'west_queue': np.mean([m['west']['queue_length'] for m in recent_metrics])
        },
        'episode': simulation_state['episode'],
        'step': simulation_state['step']
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get metrics history for charts"""
    limit = request.args.get('limit', 200, type=int)
    history = simulation_state['metrics_history'][-limit:]
    
    return jsonify({
        'timestamps': [m['step'] for m in history],
        'queue_lengths': [m['total_queue_length'] for m in history],
        'throughputs': [m['total_throughput'] for m in history],
        'phases': [m['current_phase'] for m in history],
        'north_queues': [m['north']['queue_length'] for m in history],
        'south_queues': [m['south']['queue_length'] for m in history],
        'east_queues': [m['east']['queue_length'] for m in history],
        'west_queues': [m['west']['queue_length'] for m in history],
        'rewards': [m.get('reward', 0) for m in history]
    })


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset simulation"""
    config = request.json if request.json else DEFAULT_CONFIG
    initialize_simulation(config)
    
    return jsonify({'status': 'success', 'message': 'Simulation reset'})


@app.route('/api/compare', methods=['POST'])
def compare_algorithms():
    """Compare RL vs Fixed timing over 500 steps"""
    env = simulation_state['env']
    agent = simulation_state['agent']
    
    if env is None or agent is None:
        return jsonify({'status': 'error', 'message': 'Simulation not initialized'}), 400
    
    rl_metrics = []
    fixed_metrics = []
    
    # Test RL agent
    env.reset()
    for _ in range(500):
        if simulation_state['use_cnn']:
            state = env.get_grid_observation()
        else:
            state = env.get_observation()
        
        action = agent.act(state, training=False)
        _, reward, _ = env.step(action)
        metrics = env.get_metrics()
        metrics['reward'] = reward
        rl_metrics.append(metrics)
    
    # Test fixed timing
    env.reset()
    for _ in range(500):
        action = 0 if env.time_in_phase < env.green_duration - 5 else 1
        _, reward, _ = env.step(action)
        metrics = env.get_metrics()
        metrics['reward'] = reward
        fixed_metrics.append(metrics)
    
    improvement = {
        'queue': (
            (np.mean([m['total_queue_length'] for m in fixed_metrics]) - 
             np.mean([m['total_queue_length'] for m in rl_metrics])) / 
            np.mean([m['total_queue_length'] for m in fixed_metrics]) * 100
        ),
        'throughput': (
            (np.mean([m['total_throughput'] for m in rl_metrics]) - 
             np.mean([m['total_throughput'] for m in fixed_metrics])) / 
            np.mean([m['total_throughput'] for m in fixed_metrics]) * 100
        )
    }
    
    return jsonify({
        'rl': {
            'avg_queue': float(np.mean([m['total_queue_length'] for m in rl_metrics])),
            'avg_throughput': float(np.mean([m['total_throughput'] for m in rl_metrics])),
            'total_reward': float(sum([m['reward'] for m in rl_metrics]))
        },
        'fixed': {
            'avg_queue': float(np.mean([m['total_queue_length'] for m in fixed_metrics])),
            'avg_throughput': float(np.mean([m['total_throughput'] for m in fixed_metrics])),
            'total_reward': float(sum([m['reward'] for m in fixed_metrics]))
        },
        'improvement': improvement
    })


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    return jsonify({
        'use_rl': simulation_state['use_rl'],
        'use_cnn': simulation_state['use_cnn'],
        'use_dueling': simulation_state['use_dueling'],
        'episode': simulation_state['episode'],
        'step': simulation_state['step'],
        'uptime': str(datetime.now() - simulation_state['start_time']) if simulation_state['start_time'] else 'N/A'
    })


@app.route('/api/agent-metrics', methods=['GET'])
def get_agent_metrics():
    """Get agent training metrics"""
    agent = simulation_state['agent']
    
    if agent is None:
        return jsonify({'status': 'error', 'message': 'Agent not initialized'}), 400
    
    if hasattr(agent, 'get_training_metrics'):
        return jsonify(agent.get_training_metrics())
    else:
        return jsonify({
            'total_steps': agent.steps,
            'epsilon': float(agent.epsilon),
            'loss_history': agent.loss_history[-100:],
            'reward_history': agent.reward_history[-100:]
        })


@app.route('/api/export-metrics', methods=['GET'])
def export_metrics():
    """Export metrics as JSON"""
    visualizer.export_metrics_to_json(
        simulation_state['metrics_history'],
        'data/training_logs/metrics_export.json'
    )
    
    return jsonify({
        'status': 'success',
        'message': 'Metrics exported',
        'file': 'data/training_logs/metrics_export.json'
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'simulation_running': simulation_state['running']
    })


@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start a background training process with a defined parameter space."""
    global training_manager
    if training_manager and training_manager.get_status()['status'] == 'running':
        return jsonify({'status': 'error', 'message': 'Training is already in progress.'}), 400

    # Define the hyperparameter space for the training
    parameter_space = {
        'north_traffic': [0.1, 0.3, 0.5],
        'south_traffic': [0.1, 0.3, 0.5],
        'east_traffic': [0.1, 0.3, 0.5],
        'west_traffic': [0.1, 0.3, 0.5],
        'green_duration': [20, 30, 40]
    }
    
    episodes = request.json.get('episodes', 5) if request.json else 5
    steps = request.json.get('steps', 200) if request.json else 200

    training_manager = start_training_in_background(parameter_space, episodes, steps)
    
    return jsonify({
        'status': 'success',
        'message': 'Started background training process.',
        'total_configurations': training_manager.get_status()['total_configs']
    })

@app.route('/api/training/status', methods=['GET'])
def training_status():
    """Get the status of the current training process."""
    if not training_manager:
        return jsonify({'status': 'idle', 'message': 'No training process has been started.'})
    
    return jsonify(training_manager.get_status())

@app.route('/api/training/results', methods=['GET'])
def training_results():
    """Get the results of the completed training process."""
    if not training_manager or training_manager.get_status()['status'] != 'completed':
        return jsonify({'status': 'error', 'message': 'Training is not completed or was not started.'}), 400
    
    results = training_manager.load_results()
    return jsonify({
        'status': 'success',
        'results': results
    })


@app.route('/api/', methods=['GET'])
def api_documentation():
    """Provide a list of available API endpoints."""
    endpoints = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint != 'static' and rule.methods is not None:
            methods = ','.join(sorted(rule.methods))
            endpoints.append({
                "rule": rule.rule,
                "methods": methods,
                "description": app.view_functions[rule.endpoint].__doc__
            })
    return jsonify({
        "message": "Welcome to the Smart Traffic Signal Optimizer API",
        "endpoints": endpoints
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Smart Traffic Signal Optimizer - Backend Server")
    print("="*60)
    print("Starting Flask API server...")
    print("API Documentation: http://localhost:5000/api/")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
