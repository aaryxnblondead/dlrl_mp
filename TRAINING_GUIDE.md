# Training the Traffic Signal DQN Agent

## Quick Start

Train the model with default settings:
```bash
python train_model.py
```

## Command Line Options

### Basic Training
```bash
# Train for 500 episodes with 1000 steps each (default)
python train_model.py

# Train for 1000 episodes
python train_model.py --episodes 1000

# Train with 2000 steps per episode
python train_model.py --steps 2000
```

### Advanced Options
```bash
# Use CNN-based agent (for grid observations)
python train_model.py --use-cnn

# Save checkpoints every 25 episodes
python train_model.py --save-every 25

# Full custom training
python train_model.py --episodes 1000 --steps 2000 --use-cnn --save-every 50
```

## Training Configuration

- **State Size**: 44 features (enhanced observation)
  - 12 raw queue lengths
  - 12 normalized queue lengths
  - 12 max wait times
  - 4 per-direction total queues
  - 4 phase indicators

- **Actions**: 5 possible actions
  - 0: Keep current phase
  - 1: Switch to NS Straight Green
  - 2: Switch to NS Left Green
  - 3: Switch to EW Straight Green
  - 4: Switch to EW Left Green

- **Reward System**: Prioritizes low queues
  - Strong rewards for keeping queues <10
  - Early penalties for queues >10
  - Exponential penalties for queues >15
  - Catastrophic penalties for queues >20

- **Learning Parameters**:
  - Learning Rate: 0.0003
  - Discount Factor (γ): 0.75 (prioritizes immediate rewards)
  - Epsilon Decay: 0.9995
  - Initial Epsilon: 1.0 → Final: 0.05
  - Batch Size: 128
  - Replay Buffer: 50,000 experiences

## Output Files

Training saves files to `models/` directory:

1. **Best Model**: `agent_best_cnn{False/True}.keras`
   - Saved when average queue reaches new low

2. **Final Model**: `agent_final_YYYYMMDD_HHMMSS_cnn{False/True}.keras`
   - Saved at end of training

3. **Checkpoints**: `agent_ep{N}_cnn{False/True}.keras`
   - Saved every N episodes

4. **Metrics**: `training_metrics_YYYYMMDD_HHMMSS_cnn{False/True}.json`
   - Episode rewards, queues, throughput, losses

## Monitoring Training

Progress is logged every 10 episodes:
```
Episode  10/500 | Reward:   -245.3 | Avg Queue:  15.2 | Max Queue:  42 | Throughput:   387 | ε: 0.9900 | Loss: 1.2345
Episode  20/500 | Reward:   -189.7 | Avg Queue:  12.4 | Max Queue:  35 | Throughput:   425 | ε: 0.9801 | Loss: 0.9876
```

## Example Training Sessions

### Quick Test (50 episodes)
```bash
python train_model.py --episodes 50 --steps 500
```
⏱️ ~5-10 minutes

### Standard Training (500 episodes)
```bash
python train_model.py
```
⏱️ ~30-60 minutes

### Intensive Training (1000 episodes)
```bash
python train_model.py --episodes 1000 --steps 2000
```
⏱️ ~2-3 hours

### CNN Training
```bash
python train_model.py --use-cnn --episodes 500
```
⏱️ ~45-90 minutes (slower due to CNN)

## Stopping Training

- Press `Ctrl+C` to stop training gracefully
- Models are automatically saved at checkpoints
- Resume training by loading a checkpoint model

## Using Trained Models

After training, load the best model in your application:

```python
from dqn_agent import DQNAgent

agent = DQNAgent(num_actions=5, state_size=44)
agent.q_network = keras.models.load_model('models/agent_best_cnnFalse.keras')
agent.epsilon = 0.01  # Use low epsilon for exploitation
```

## Troubleshooting

**Out of Memory**:
- Reduce `--steps` parameter
- Reduce batch size in dqn_agent.py

**Training Too Slow**:
- Start with fewer episodes: `--episodes 100`
- Reduce steps: `--steps 500`
- Use MLP instead of CNN (remove `--use-cnn`)

**Poor Performance**:
- Increase training episodes: `--episodes 1000`
- Check that reward function in traffic_env.py is correct
- Verify queue penalties are high enough
