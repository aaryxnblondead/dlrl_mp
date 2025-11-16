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
    # New structure for arrival rates: nested dictionary
    arrival_rates = {
        'north': {
            'left': config.get('north_left', 0.1),
            'straight': config.get('north_straight', 0.2),
            'right': config.get('north_right', 0.1)
        },
        'south': {
            'left': config.get('south_left', 0.1),
            'straight': config.get('south_straight', 0.2),
            'right': config.get('south_right', 0.1)
        },
        'east': {
            'left': config.get('east_left', 0.08),
            'straight': config.get('east_straight', 0.15),
            'right': config.get('east_right', 0.08)
        },
        'west': {
            'left': config.get('west_left', 0.08),
            'straight': config.get('west_straight', 0.15),
            'right': config.get('west_right', 0.08)
        }
    }
    
    simulation_state['env'] = TrafficIntersection(
        arrival_rates=arrival_rates,
        min_green=config.get('min_green', 10),
        yellow_duration=config.get('yellow_duration', 4)
    )
    
    # Initialize agent based on configuration
    use_cnn = config.get('use_cnn', False)
    
    # The new environment has 5 actions: 0 (keep), 1-4 (switch to a green phase)
    # The new state size is 28 for MLP, and the grid is 4x3 for CNN
    agent = DQNAgent(
        num_actions=5,
        state_size=28,
        use_cnn=use_cnn,
        **DQN_CONFIG
    )
    
    simulation_state['agent'] = agent
    simulation_state['use_cnn'] = use_cnn
    
    # Model loading logic remains the same, but ensure models are compatible
    try:
        model_files = list(MODELS_DIR.glob('agent_model_*.h5'))
        if model_files:
            model_path = sorted(model_files)[-1] # Load the latest model
            agent.load_model(str(model_path))
            print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Warning: Could not load pre-trained model: {e}")
    
    simulation_state['metrics_history'] = []
    simulation_state['agent_metrics'] = {
        'loss_history': [],
        'reward_history': []
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
        # Fixed timing: cycle through phases with fixed green time
        # This is a simplified fixed controller for the new 8-phase system
        green_duration = 15 # Fixed green time for non-RL mode
        is_green_phase = env.current_phase % 2 == 0
        
        if is_green_phase and env.time_in_phase >= green_duration:
            # Request switch to the next green phase in sequence
            current_green_phase_idx = env.current_phase // 2
            next_green_phase_idx = (current_green_phase_idx + 1) % 4
            action = next_green_phase_idx + 1 # Actions 1-4 map to phases 0,2,4,6
        else:
            action = 0 # Keep current phase
    
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
        # If no history, return the initial state of the environment
        env = simulation_state.get('env')
        if env:
            initial_metrics = env.get_metrics()
            return jsonify({
                'current': initial_metrics,
                'averages': {}, # No averages yet
                'episode': simulation_state['episode'],
                'step': simulation_state['step']
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Simulation not initialized'
            }), 400

    latest = simulation_state['metrics_history'][-1]
    
    # Calculate averages over last 50 steps
    recent_metrics = simulation_state['metrics_history'][-50:]
    
    # Calculate average queue lengths for each lane
    lanes_avg = {f"{d}_{t}": np.mean([m['lanes'][d][t]['queue_length'] for m in recent_metrics])
                 for d in ['north', 'south', 'east', 'west'] for t in ['left', 'straight', 'right']}

    return jsonify({
        'current': latest,
        'averages': {
            'queue_length': np.mean([m['total_queue_length'] for m in recent_metrics]),
            'throughput': np.mean([m['total_throughput'] for m in recent_metrics]),
            **lanes_avg
        },
        'episode': simulation_state['episode'],
        'step': simulation_state['step']
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get metrics history for charts"""
    limit = request.args.get('limit', 200, type=int)
    history = simulation_state['metrics_history'][-limit:]
    
    # New structure for lane data
    lanes_history = {f"{d}_{t}": [] for d in ['north', 'south', 'east', 'west'] for t in ['left', 'straight', 'right']}
    
    for m in history:
        for d in ['north', 'south', 'east', 'west']:
            for t in ['left', 'straight', 'right']:
                lanes_history[f"{d}_{t}"].append(m['lanes'][d][t]['queue_length'])

    return jsonify({
        'timestamps': [m['step'] for m in history],
        'queue_lengths': [m['total_queue_length'] for m in history],
        'throughputs': [m['total_throughput'] for m in history],
        'phases': [m['current_phase_name'] for m in history], # Use name for clarity
        'rewards': [m.get('reward', 0) for m in history],
        'lanes': lanes_history
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

    training_mode = request.json.get('mode', 'quick') if request.json else 'quick'
    
    if training_mode == 'comprehensive':
        # COMPREHENSIVE MODE: All 12 lanes individually configured
        # Traffic density levels: Low (0.05), Medium-Low (0.15), Medium (0.25), Medium-High (0.35), High (0.5)
        # This creates diverse scenarios for CNN to learn complex patterns
        traffic_levels = [0.05, 0.15, 0.25, 0.35, 0.5]
        parameter_space = {
            'north_left': traffic_levels,
            'north_straight': traffic_levels,
            'north_right': traffic_levels,
            'south_left': traffic_levels,
            'south_straight': traffic_levels,
            'south_right': traffic_levels,
            'east_left': traffic_levels,
            'east_straight': traffic_levels,
            'east_right': traffic_levels,
            'west_left': traffic_levels,
            'west_straight': traffic_levels,
            'west_right': traffic_levels,
            'min_green': [10, 15, 20, 25, 30]  # Min green signal duration
        }
        # Total configurations: 5^13 = 1,220,703,125 (too many!)
        # We'll use random sampling instead
    else:
        # QUICK MODE: Simplified per-direction training (original approach)
        parameter_space = {
            'north_traffic': [0.1, 0.3, 0.5],
            'south_traffic': [0.1, 0.3, 0.5],
            'east_traffic': [0.1, 0.3, 0.5],
            'west_traffic': [0.1, 0.3, 0.5],
            'min_green': [10, 20, 30]
        }
        # Total configurations: 3^5 = 243
    
    episodes = request.json.get('episodes', 5) if request.json else 5
    steps = request.json.get('steps', 200) if request.json else 200
    
    # For comprehensive mode, use sampling instead of exhaustive search
    if training_mode == 'comprehensive':
        num_samples = request.json.get('num_samples', 1000) if request.json else 1000
        training_manager = start_training_in_background(parameter_space, episodes, steps, sample_configs=num_samples)
        total_configs = num_samples
    else:
        training_manager = start_training_in_background(parameter_space, episodes, steps)
        total_configs = 243  # 3^5
    
    return jsonify({
        'status': 'success',
        'message': 'Started background training process.',
        'mode': training_mode,
        'total_configurations': total_configs
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
