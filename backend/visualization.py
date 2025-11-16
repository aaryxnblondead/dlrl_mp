# backend/visualization.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from io import BytesIO
import json
from pathlib import Path
import plotly.graph_objs as go
from plotly.subplots import make_subplots

class TrainingVisualizer:
    """
    Visualization utilities for training progress and simulation metrics
    Supports both static and interactive charts
    """
    
    def __init__(self, log_dir: str = 'data/training_logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def plot_training_metrics(metrics_history: list, save_path: str = None) -> Figure:
        """
        Create comprehensive training visualization
        
        Args:
            metrics_history: List of metrics dictionaries from each step
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
        
        # Extract metrics
        episodes = range(len(metrics_history))
        rewards = [m.get('reward', 0) for m in metrics_history]
        queue_lengths = [m.get('total_queue_length', 0) for m in metrics_history]
        throughputs = [m.get('total_throughput', 0) for m in metrics_history]
        
        # Plot 1: Rewards over time
        axes[0, 0].plot(episodes, rewards, 'b-', alpha=0.6, linewidth=1)
        axes[0, 0].set_title('Rewards per Step')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot moving average
        if len(rewards) > 50:
            moving_avg = np.convolve(rewards, np.ones(50)/50, mode='valid')
            axes[0, 0].plot(range(50, 50+len(moving_avg)), moving_avg, 'r-', linewidth=2, label='50-step MA')
            axes[0, 0].legend()
        
        # Plot 2: Queue lengths
        axes[0, 1].plot(episodes, queue_lengths, 'g-', alpha=0.6, linewidth=1)
        axes[0, 1].set_title('Total Queue Length')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Queue Length')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Throughput
        axes[1, 0].plot(episodes, throughputs, 'purple', alpha=0.6, linewidth=1)
        axes[1, 0].set_title('Total Throughput')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Vehicles Processed')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Reward distribution
        axes[1, 1].hist(rewards, bins=50, color='cyan', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_episode_comparison(rl_metrics: dict, fixed_metrics: dict, save_path: str = None) -> Figure:
        """
        Create comparison plot between RL and fixed timing algorithms
        
        Args:
            rl_metrics: Metrics from RL agent
            fixed_metrics: Metrics from fixed timing
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('RL vs Fixed Timing Algorithm Comparison', fontsize=16, fontweight='bold')
        
        algorithms = ['RL Agent', 'Fixed Timing']
        colors = ['#00ff00', '#ff0000']
        
        # Comparison data
        rl_vals = [
            rl_metrics.get('avg_queue', 0),
            rl_metrics.get('avg_throughput', 0),
            rl_metrics.get('total_reward', 0)
        ]
        
        fixed_vals = [
            fixed_metrics.get('avg_queue', 0),
            fixed_metrics.get('avg_throughput', 0),
            fixed_metrics.get('total_reward', 0)
        ]
        
        metrics_names = ['Avg Queue Length', 'Avg Throughput', 'Total Reward']
        
        for idx, (ax, metric_name, rl_val, fixed_val) in enumerate(
            zip(axes, metrics_names, rl_vals, fixed_vals)
        ):
            bars = ax.bar(algorithms, [rl_val, fixed_val], color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_lane_metrics(lane_metrics: dict, save_path: str = None) -> Figure:
        """
        Create visualization of metrics for each lane
        
        Args:
            lane_metrics: Dictionary with lane-specific metrics
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Per-Lane Traffic Metrics', fontsize=16, fontweight='bold')
        
        lanes = ['north', 'south', 'east', 'west']
        axes_flat = axes.flatten()
        
        for ax, lane in zip(axes_flat, lanes):
            if lane in lane_metrics:
                metrics = lane_metrics[lane]
                
                categories = ['Queue', 'Throughput', 'Avg Wait']
                values = [
                    metrics.get('queue_length', 0),
                    metrics.get('throughput', 0),
                    metrics.get('avg_wait_time', 0)
                ]
                
                bars = ax.bar(categories, values, color=['#ff6b6b', '#4ecdc4', '#45b7d1'], alpha=0.7, edgecolor='black')
                ax.set_title(f'{lane.upper()} Lane', fontsize=11, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def create_interactive_dashboard(metrics_history: list) -> go.Figure:
        """
        Create interactive Plotly dashboard for real-time monitoring
        
        Args:
            metrics_history: List of metrics dictionaries
        
        Returns:
            Plotly figure with subplots
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Queue Length Over Time',
                'Throughput Over Time',
                'Rewards Over Time',
                'Phase Distribution',
                'Lane Comparison',
                'Metrics Summary'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'pie'}],
                [{'type': 'bar'}, {'type': 'indicator'}]
            ]
        )
        
        # Extract data
        steps = list(range(len(metrics_history)))
        queue_lengths = [m.get('total_queue_length', 0) for m in metrics_history]
        throughputs = [m.get('total_throughput', 0) for m in metrics_history]
        rewards = [m.get('reward', 0) for m in metrics_history]
        phases = [m.get('current_phase', 'NORTH_SOUTH') for m in metrics_history]
        
        # Plot 1: Queue length
        fig.add_trace(
            go.Scatter(x=steps, y=queue_lengths, mode='lines', name='Queue Length',
                      line=dict(color='#ff6b6b', width=2)),
            row=1, col=1
        )
        
        # Plot 2: Throughput
        fig.add_trace(
            go.Scatter(x=steps, y=throughputs, mode='lines', name='Throughput',
                      line=dict(color='#4ecdc4', width=2)),
            row=1, col=2
        )
        
        # Plot 3: Rewards
        fig.add_trace(
            go.Scatter(x=steps, y=rewards, mode='lines', name='Reward',
                      line=dict(color='#45b7d1', width=2)),
            row=2, col=1
        )
        
        # Plot 4: Phase pie chart
        phase_counts = {}
        for phase in phases:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        fig.add_trace(
            go.Pie(labels=list(phase_counts.keys()), values=list(phase_counts.values()),
                   name='Phases'),
            row=2, col=2
        )
        
        # Plot 5: Lane comparison (last metrics)
        if metrics_history:
            last_metrics = metrics_history[-1]
            lanes = ['north', 'south', 'east', 'west']
            lane_queues = [last_metrics.get(lane, {}).get('queue_length', 0) for lane in lanes]
            
            fig.add_trace(
                go.Bar(x=lanes, y=lane_queues, name='Queue by Lane',
                      marker=dict(color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text='Traffic Signal Optimization - Interactive Dashboard',
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def export_metrics_to_json(metrics_history: list, filepath: str):
        """Export metrics to JSON for external analysis"""
        with open(filepath, 'w') as f:
            json.dump(metrics_history, f, indent=2)
    
    @staticmethod
    def plot_agent_learning_curves(agent_metrics: dict, save_path: str = None) -> Figure:
        """
        Plot DQN agent learning curves (loss, Q-values, TD error)
        
        Args:
            agent_metrics: Dictionary with agent training metrics
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('DQN Agent Learning Curves', fontsize=16, fontweight='bold')
        
        steps = range(len(agent_metrics.get('loss_history', [])))
        
        # Plot 1: Loss
        if 'loss_history' in agent_metrics:
            axes[0].plot(steps, agent_metrics['loss_history'], 'b-', alpha=0.6, linewidth=1)
            if len(agent_metrics['loss_history']) > 100:
                moving_avg = np.convolve(agent_metrics['loss_history'], np.ones(100)/100, mode='valid')
                axes[0].plot(range(100, 100+len(moving_avg)), moving_avg, 'r-', linewidth=2, label='100-step MA')
            axes[0].set_title('Training Loss')
            axes[0].set_xlabel('Update Step')
            axes[0].set_ylabel('Loss (MSE)')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
        
        # Plot 2: Q-values
        if 'q_value_history' in agent_metrics:
            axes[1].plot(steps[:len(agent_metrics['q_value_history'])], 
                        agent_metrics['q_value_history'], 'g-', alpha=0.6, linewidth=1)
            axes[1].set_title('Max Q-values')
            axes[1].set_xlabel('Step')
            axes[1].set_ylabel('Max Q-value')
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: TD Error
        if 'td_error_history' in agent_metrics:
            axes[2].plot(steps[:len(agent_metrics['td_error_history'])], 
                        agent_metrics['td_error_history'], 'purple', alpha=0.6, linewidth=1)
            axes[2].set_title('Temporal Difference Error')
            axes[2].set_xlabel('Update Step')
            axes[2].set_ylabel('Avg TD Error')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
