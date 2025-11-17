# backend/dqn_agent.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
from typing import Optional

class DQNNetwork(keras.Model):
    """Deep Q-Network for traffic signal control - Enhanced Architecture"""
    def __init__(self, num_actions: int = 5):
        super(DQNNetwork, self).__init__()
        
        # Deeper network with better capacity for complex traffic patterns
        self.dense1 = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001))
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.3)
        
        self.dense2 = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001))
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.3)
        
        self.dense3 = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001))
        self.bn3 = layers.BatchNormalization()
        self.dropout3 = layers.Dropout(0.2)
        
        self.dense4 = layers.Dense(64, activation='relu')
        
        # Dueling DQN architecture - separate value and advantage streams
        self.value_stream = layers.Dense(32, activation='relu')
        self.value = layers.Dense(1)
        
        self.advantage_stream = layers.Dense(32, activation='relu')
        self.advantages = layers.Dense(num_actions)
    
    def call(self, state, training=False):
        """Forward pass with Dueling DQN architecture"""
        x = self.dense1(state)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        
        x = self.dense3(x)
        x = self.bn3(x, training=training)
        x = self.dropout3(x, training=training)
        
        x = self.dense4(x)
        
        # Dueling streams
        value = self.value_stream(x)
        value = self.value(value)
        
        advantages = self.advantage_stream(x)
        advantages = self.advantages(advantages)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - tf.reduce_mean(advantages, axis=1, keepdims=True))
        
        return q_values

class DQNAgentWithCNN(keras.Model):
    """DQN Agent with CNN for grid-based observation - Enhanced for Traffic Patterns"""
    def __init__(self, num_actions: int = 5):
        super(DQNAgentWithCNN, self).__init__()
        
        # Multi-scale CNN to capture different traffic patterns
        # First conv layer - local patterns
        self.conv1 = layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same',
                                   kernel_regularizer=keras.regularizers.l2(0.0001))
        self.bn1 = layers.BatchNormalization()
        
        # Second conv layer - broader patterns
        self.conv2 = layers.Conv2D(128, kernel_size=(2, 2), activation='relu', padding='same',
                                   kernel_regularizer=keras.regularizers.l2(0.0001))
        self.bn2 = layers.BatchNormalization()
        
        # Third conv layer - full context
        self.conv3 = layers.Conv2D(256, kernel_size=(3, 2), activation='relu', padding='same',
                                   kernel_regularizer=keras.regularizers.l2(0.0001))
        self.bn3 = layers.BatchNormalization()
        
        self.flatten = layers.Flatten()
        
        # Dense layers for decision making
        self.dense1 = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001))
        self.dropout1 = layers.Dropout(0.3)
        self.dense2 = layers.Dense(256, activation='relu')
        self.dropout2 = layers.Dropout(0.3)
        self.dense3 = layers.Dense(128, activation='relu')
        
        # Dueling DQN for CNN
        self.value_stream = layers.Dense(64, activation='relu')
        self.value = layers.Dense(1)
        
        self.advantage_stream = layers.Dense(64, activation='relu')
        self.advantages = layers.Dense(num_actions)
    
    def call(self, grid_state, training=False):
        """Forward pass with grid input and Dueling architecture"""
        # Add batch and channel dimensions if needed
        if len(grid_state.shape) == 2:
            grid_state = tf.expand_dims(grid_state, axis=0)
        if len(grid_state.shape) == 3:
            grid_state = tf.expand_dims(grid_state, axis=-1)
        
        x = self.conv1(grid_state)
        x = self.bn1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        
        x = self.flatten(x)
        
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        
        # Dueling streams
        value = self.value_stream(x)
        value = self.value(value)
        
        advantages = self.advantage_stream(x)
        advantages = self.advantages(advantages)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - tf.reduce_mean(advantages, axis=1, keepdims=True))
        
        return q_values

class DQNAgent:
    """DQN Agent with experience replay and target network"""
    def __init__(self, 
                 num_actions: int = 5,
                 state_size: int = 44,  # Updated for enhanced state observation
                 learning_rate: float = 0.0003,  # Lower learning rate for stability
                 gamma: float = 0.75,  # Lower gamma for immediate queue clearing (prioritize short-term)
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,  # Higher minimum exploration
                 epsilon_decay: float = 0.9995,  # Slower decay
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
        
        # Enhanced replay buffer
        self.memory = deque(maxlen=50000)  # Larger memory for more diverse experiences
        self.batch_size = 128  # Larger batch for stable learning
        self.update_frequency = 2  # Update more frequently
        self.target_update_frequency = 500  # Update target network every 500 steps
        self.steps = 0
        
        # Prioritized experience replay weights
        self.priority_scale = 0.6  # How much prioritization to use
        
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
            self.q_network.build((None, 4, 3, 1)) # Updated shape
            self.target_network.build((None, 4, 3, 1)) # Updated shape
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
        """
        if training and np.random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # Predict Q-values
        if self.use_cnn:
            # Shape is already (4,3) from env, needs batch and channel
            if len(state.shape) == 2:
                state = state[np.newaxis, :, :, np.newaxis] # (1, 4, 3, 1)
            elif len(state.shape) == 3: # (4, 3, 1)
                 state = state[np.newaxis, :, :, :] # (1, 4, 3, 1)
        else:
            # Shape is already (28,), needs batch
            state = state[np.newaxis, :]
        
        q_values = self.q_network(state, training=False)
        return int(np.argmax(q_values[0].numpy()))
    
    def replay(self, batch_size: Optional[int] = None) -> float:
        """
        Experience replay - train on batch of experiences
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
            # states are (batch, 4, 3), need to add channel dim
            if len(states.shape) == 3:
                states = states[:, :, :, np.newaxis]
                next_states = next_states[:, :, :, np.newaxis]
        
        # Double DQN: use main network to select action, target network to evaluate
        q_values_next = self.q_network(next_states, training=False).numpy()
        best_actions = np.argmax(q_values_next, axis=1)
        
        target_q_values_next = self.target_network(next_states, training=False).numpy()
        target_q_values = rewards + self.gamma * target_q_values_next[np.arange(batch_size), best_actions] * (1 - dones)
        
        # Train main network
        with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)
            
            # Gather Q-values for taken actions
            action_masks = tf.one_hot(actions, self.num_actions)
            q_values_taken = tf.reduce_sum(q_values * action_masks, axis=1)
            
            # Huber loss for stability (less sensitive to outliers than MSE)
            loss = keras.losses.Huber()(target_q_values, q_values_taken)
        
        # Backpropagation with gradient clipping
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        clipped_gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        self.optimizer.apply_gradients(zip(clipped_gradients, self.q_network.trainable_variables))
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_frequency == 0:
            self.update_target_network()
        
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