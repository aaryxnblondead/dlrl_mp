# backend/dqn_cnn_agent.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
from typing import Tuple

class CNNDQNNetwork(keras.Model):
    """
    Advanced CNN-based DQN Network for grid observations
    Reference: Mnih et al. (2015) "Human-Level Control through Deep Reinforcement Learning"
    
    Architecture:
    - Input: 4x4 grid representation
    - Conv2D layers for spatial feature extraction
    - Dense layers for decision making
    - Output: Q-values for each action
    """
    def __init__(self, num_actions: int = 2):
        super(CNNDQNNetwork, self).__init__()
        
        # Input: (batch, 4, 4, 1)
        self.conv1 = layers.Conv2D(
            filters=32, 
            kernel_size=2, 
            strides=1,
            padding='same',
            activation='relu',
            name='conv1'
        )
        
        self.conv2 = layers.Conv2D(
            filters=64,
            kernel_size=2,
            strides=1,
            padding='same',
            activation='relu',
            name='conv2'
        )
        
        # Global pooling to flatten spatial dimensions
        self.pool = layers.GlobalAveragePooling2D(name='global_pool')
        
        # Dense layers
        self.dense1 = layers.Dense(128, activation='relu', name='dense1')
        self.batch_norm1 = layers.BatchNormalization(name='batch_norm1')
        self.dropout1 = layers.Dropout(0.3, name='dropout1')
        
        self.dense2 = layers.Dense(64, activation='relu', name='dense2')
        self.batch_norm2 = layers.BatchNormalization(name='batch_norm2')
        self.dropout2 = layers.Dropout(0.3, name='dropout2')
        
        # Output: Q-values for each action
        self.q_values = layers.Dense(num_actions, name='q_values')
    
    def call(self, grid_state, training=False):
        """
        Forward pass with grid input
        
        Args:
            grid_state: Input grid (batch_size, 4, 4, 1)
            training: Whether in training mode for dropout/batchnorm
        
        Returns:
            Q-values (batch_size, num_actions)
        """
        # CNN feature extraction
        x = self.conv1(grid_state)
        x = self.conv2(x)
        x = self.pool(x)
        
        # Dense layers with regularization
        x = self.dense1(x)
        x = self.batch_norm1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.batch_norm2(x, training=training)
        x = self.dropout2(x, training=training)
        
        # Q-value output
        q_values = self.q_values(x)
        
        return q_values


class DuelingCNNDQNNetwork(keras.Model):
    """
    Dueling DQN Architecture
    Reference: Wang et al. (2015) "Dueling Network Architectures for Deep Reinforcement Learning"
    
    Separates value and advantage streams for more stable learning
    """
    def __init__(self, num_actions: int = 2):
        super(DuelingCNNDQNNetwork, self).__init__()
        
        # Shared CNN layers
        self.conv1 = layers.Conv2D(
            filters=32, kernel_size=2, strides=1, padding='same',
            activation='relu', name='shared_conv1'
        )
        
        self.conv2 = layers.Conv2D(
            filters=64, kernel_size=2, strides=1, padding='same',
            activation='relu', name='shared_conv2'
        )
        
        self.pool = layers.GlobalAveragePooling2D(name='shared_pool')
        
        # Value stream
        self.value_dense1 = layers.Dense(64, activation='relu', name='value_dense1')
        self.value_output = layers.Dense(1, name='value_output')
        
        # Advantage stream
        self.advantage_dense1 = layers.Dense(64, activation='relu', name='advantage_dense1')
        self.advantage_output = layers.Dense(num_actions, name='advantage_output')
        
        self.num_actions = num_actions
    
    def call(self, grid_state, training=False):
        """
        Forward pass with dueling architecture
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        """
        # Shared feature extraction
        x = self.conv1(grid_state)
        x = self.conv2(x)
        x = self.pool(x)
        
        # Value stream
        value = self.value_dense1(x)
        value = self.value_output(value)
        
        # Advantage stream
        advantage = self.advantage_dense1(x)
        advantage = self.advantage_output(advantage)
        
        # Combine value and advantage
        # Normalize advantages by subtracting mean
        advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
        advantage_normalized = advantage - advantage_mean
        
        # Q-values
        q_values = value + advantage_normalized
        
        return q_values


class DoubleDQNAgent:
    """
    Double DQN Agent with CNN support
    Reference: Van Hasselt et al. (2015) "Deep Reinforcement Learning with Double Q-learning"
    
    Key improvements:
    - Dueling architecture for better feature learning
    - Double Q-learning to reduce overestimation
    - Prioritized experience replay ready
    - Proper target network updates
    """
    def __init__(self,
                 num_actions: int = 2,
                 use_cnn: bool = True,
                 dueling: bool = False,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 2000,
                 batch_size: int = 32,
                 update_frequency: int = 4,
                 target_update_frequency: int = 1000):
        """
        Args:
            num_actions: Number of possible actions
            use_cnn: Whether to use CNN (True) or dense network
            dueling: Whether to use dueling architecture
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay per episode
            memory_size: Size of replay buffer
            batch_size: Batch size for training
            update_frequency: How often to run a training step
            target_update_frequency: How often to update the target network
        """
        self.num_actions = num_actions
        self.use_cnn = use_cnn
        self.dueling = dueling
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency  # Steps between target network updates
        
        # Replay buffer
        self.memory = deque(maxlen=memory_size)
        self.steps = 0
        self.updates = 0
        
        # Build networks
        if use_cnn:
            if dueling:
                self.q_network = DuelingCNNDQNNetwork(num_actions)
                self.target_network = DuelingCNNDQNNetwork(num_actions)
            else:
                self.q_network = CNNDQNNetwork(num_actions)
                self.target_network = CNNDQNNetwork(num_actions)
        else:
            # Fallback to dense for non-CNN (from previous phase)
            from dqn_agent import DQNNetwork
            self.q_network = DQNNetwork(num_actions)
            self.target_network = DQNNetwork(num_actions)
        
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Build networks with proper shapes
        if use_cnn:
            self.q_network.build((None, 4, 4, 1))
            self.target_network.build((None, 4, 4, 1))
        else:
            self.q_network.build((None, 13))
            self.target_network.build((None, 13))
        
        # Copy weights to target network
        self.update_target_network()
        
        # Metrics
        self.loss_history = []
        self.reward_history = []
        self.q_value_history = []
        self.td_error_history = []
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Epsilon-greedy action selection
        
        Args:
            state: Current observation
            training: Whether in training mode (exploration)
        
        Returns:
            Action index
        """
        if training and np.random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # Prepare state
        if self.use_cnn:
            if len(state.shape) == 2:
                state = state[np.newaxis, :, :, np.newaxis]
            else:
                state = state[np.newaxis, :]
        else:
            state = state[np.newaxis, :]
        
        # Get Q-values
        q_values = self.q_network(state, training=False)
        self.q_value_history.append(float(tf.reduce_max(q_values).numpy()))
        
        return int(np.argmax(q_values[0].numpy()))
    
    def replay(self, batch_size: int = None) -> float:
        """
        Double DQN training step with proper target calculation
        
        Returns:
            Loss value
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.memory) < batch_size:
            return 0.0
        
        # Sample minibatch
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([exp[0] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])
        
        # Prepare states
        if self.use_cnn:
            if len(states.shape) == 3:
                states = states[:, :, :, np.newaxis]
                next_states = next_states[:, :, :, np.newaxis]
        
        # Double DQN: use main network to select action, target network to evaluate
        # This reduces overestimation bias
        next_q_values_main = self.q_network(next_states, training=False).numpy()
        best_actions = np.argmax(next_q_values_main, axis=1)
        
        next_q_values_target = self.target_network(next_states, training=False).numpy()
        target_q_values = rewards + self.gamma * next_q_values_target[np.arange(batch_size), best_actions] * (1 - dones)
        
        # Train main network
        with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)
            
            # Gather Q-values for taken actions
            action_masks = tf.one_hot(actions, self.num_actions)
            q_values_taken = tf.reduce_sum(q_values * action_masks, axis=1)
            
            # TD error
            td_error = target_q_values - q_values_taken.numpy()
            self.td_error_history.append(float(np.mean(np.abs(td_error))))
            
            # Loss (MSE)
            loss = keras.losses.MSE(target_q_values, q_values_taken)
            loss = tf.reduce_mean(loss)
        
        # Backpropagation
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        self.loss_history.append(float(loss))
        self.updates += 1
        
        return float(loss)
    
    def update_target_network(self):
        """Update target network weights from main network"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath: str):
        """Save model weights"""
        self.q_network.save_weights(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model weights"""
        self.q_network.load_weights(filepath)
        self.update_target_network()
        print(f"Model loaded from {filepath}")
    
    def get_training_metrics(self) -> dict:
        """Get training metrics"""
        return {
            'total_steps': self.steps,
            'total_updates': self.updates,
            'epsilon': self.epsilon,
            'avg_loss': float(np.mean(self.loss_history[-100:])) if self.loss_history else 0,
            'avg_q_value': float(np.mean(self.q_value_history[-100:])) if self.q_value_history else 0,
            'avg_td_error': float(np.mean(self.td_error_history[-100:])) if self.td_error_history else 0
        }
