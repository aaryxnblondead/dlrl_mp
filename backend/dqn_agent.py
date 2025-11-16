# dqn_agent.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
from typing import Optional

class DQNNetwork(keras.Model):
    """Deep Q-Network for traffic signal control"""
    def __init__(self, num_actions: int = 2):
        super(DQNNetwork, self).__init__()
        
        # Input: 13 features (4 lanes × 3 features + total_waiting)
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(0.2)
        self.dense2 = layers.Dense(128, activation='relu')
        self.dropout2 = layers.Dropout(0.2)
        self.dense3 = layers.Dense(64, activation='relu')
        
        # Output layer: Q-values for each action
        self.q_values = layers.Dense(num_actions)
    
    def call(self, state, training=False):
        """Forward pass"""
        x = self.dense1(state)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        q_values = self.q_values(x)
        return q_values

class DQNAgentWithCNN(keras.Model):
    """DQN Agent with CNN for grid-based observation"""
    def __init__(self, num_actions: int = 2):
        super(DQNAgentWithCNN, self).__init__()
        
        # CNN for processing 4x4 grid
        self.conv1 = layers.Conv2D(32, kernel_size=2, activation='relu', padding='same')
        self.conv2 = layers.Conv2D(64, kernel_size=2, activation='relu', padding='same')
        self.pool = layers.GlobalAveragePooling2D()
        
        # Dense layers
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(0.2)
        self.dense2 = layers.Dense(64, activation='relu')
        
        # Q-value output
        self.q_values = layers.Dense(num_actions)
    
    def call(self, grid_state, training=False):
        """Forward pass with grid input"""
        # Add batch and channel dimensions if needed
        if len(grid_state.shape) == 2:
            grid_state = tf.expand_dims(grid_state, axis=0)
        if len(grid_state.shape) == 3:
            grid_state = tf.expand_dims(grid_state, axis=-1)
        
        x = self.conv1(grid_state)
        x = self.conv2(x)
        x = self.pool(x)
        
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        
        q_values = self.q_values(x)
        return q_values

class DQNAgent:
    """DQN Agent with experience replay and target network"""
    def __init__(self, 
                 num_actions: int = 2,
                 state_size: int = 13,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 use_cnn: bool = False):
        """
        Args:
            num_actions: Number of possible actions
            state_size: Size of state vector
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial epsilon for exploration
            epsilon_end: Minimum epsilon
            epsilon_decay: Epsilon decay rate per episode
            use_cnn: Whether to use CNN (for grid input)
        """
        self.num_actions = num_actions
        self.state_size = state_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.use_cnn = use_cnn
        
        # Replay buffer
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.update_frequency = 4  # Update network every 4 steps
        self.steps = 0
        
        # Networks
        if use_cnn:
            self.q_network = DQNAgentWithCNN(num_actions)
            self.target_network = DQNAgentWithCNN(num_actions)
        else:
            self.q_network = DQNNetwork(num_actions)
            self.target_network = DQNNetwork(num_actions)
        
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Build networks
        if use_cnn:
            self.q_network.build((None, 4, 4, 1))
            self.target_network.build((None, 4, 4, 1))
        else:
            self.q_network.build((None, state_size))
            self.target_network.build((None, state_size))
        
        # Copy weights to target network
        self.update_target_network()
        
        # Training metrics
        self.loss_history = []
        self.reward_history = []
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy strategy
        
        Args:
            state: Current state
            training: Whether in training mode (uses exploration)
        
        Returns:
            Action index
        """
        if training and np.random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # Predict Q-values
        if self.use_cnn:
            if len(state.shape) == 2:
                state = state[np.newaxis, :, :, np.newaxis]
            else:
                state = state[np.newaxis, :]
        else:
            state = state[np.newaxis, :]
        
        q_values = self.q_network(state, training=False)
        return int(np.argmax(q_values[0].numpy()))
    
    def replay(self, batch_size: Optional[int] = None) -> float:
        """
        Experience replay - train on batch of experiences
        
        Returns:
            Loss value
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.memory) < batch_size:
            return 0.0
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([exp[0] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])
        
        # Prepare states for network
        if self.use_cnn:
            if len(states.shape) == 3:
                states = states[:, :, :, np.newaxis]
                next_states = next_states[:, :, :, np.newaxis]
        
        # Compute target Q-values using target network
        target_q_values = self.target_network(next_states, training=False).numpy()
        target_q_values = rewards + self.gamma * np.max(target_q_values, axis=1) * (1 - dones)
        
        # Train main network
        with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)
            
            # Gather Q-values for taken actions
            action_masks = tf.one_hot(actions, self.num_actions)
            q_values_taken = tf.reduce_sum(q_values * action_masks, axis=1)
            
            # Compute loss
            loss = keras.losses.MeanSquaredError()(target_q_values, q_values_taken)
            loss = tf.reduce_mean(loss)
        
        # Backpropagation
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        
        # Filter out None gradients
        grads_and_vars = [(g, v) for g, v in zip(gradients, self.q_network.trainable_variables) if g is not None]
        if grads_and_vars:
            self.optimizer.apply_gradients(grads_and_vars)
        
        loss_value = float(loss.numpy())
        self.loss_history.append(loss_value)
        return loss_value
    
    def update_target_network(self):
        """Update target network with main network weights"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath: str):
        """Save model weights"""
        self.q_network.save_weights(filepath)
    
    def load_model(self, filepath: str):
        """Load model weights"""
        self.q_network.load_weights(filepath)
        self.update_target_network()
