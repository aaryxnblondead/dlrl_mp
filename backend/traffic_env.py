# backend/traffic_env.py
"""
Advanced Traffic Intersection Environment
- 12 lanes (left, straight, right for each of 4 approaches)
- 8-phase signal system (including protected left turns)
- Vehicle destinations and realistic turning movements
"""
import numpy as np
import random
from collections import deque

# Constants
DIRECTIONS = ['north', 'south', 'east', 'west']
TURNS = ['left', 'straight', 'right']
LANE_MAP = {
    'north': {'left': 0, 'straight': 1, 'right': 2},
    'south': {'left': 3, 'straight': 4, 'right': 5},
    'east': {'left': 6, 'straight': 7, 'right': 8},
    'west': {'left': 9, 'straight': 10, 'right': 11},
}
NUM_LANES = 12

# Signal Phases (NS: North-South, EW: East-West)
# Green phases are even, yellow are odd
PHASES = {
    0: 'NS_STRAIGHT_GREEN',
    1: 'NS_STRAIGHT_YELLOW',
    2: 'NS_LEFT_GREEN',
    3: 'NS_LEFT_YELLOW',
    4: 'EW_STRAIGHT_GREEN',
    5: 'EW_STRAIGHT_YELLOW',
    6: 'EW_LEFT_GREEN',
    7: 'EW_LEFT_YELLOW',
}
NUM_PHASES = len(PHASES)

# Defines which lanes have a green light for each primary phase
GREEN_PHASE_LANE_MAP = {
    0: [('north', 'straight'), ('south', 'straight'), ('north', 'right'), ('south', 'right')],
    2: [('north', 'left'), ('south', 'left')],
    4: [('east', 'straight'), ('west', 'straight'), ('east', 'right'), ('west', 'right')],
    6: [('east', 'left'), ('west', 'left')],
}

class Vehicle:
    vehicle_id_counter = 0
    
    def __init__(self, step_created, destination_turn):
        self.step_created = step_created
        self.destination_turn = destination_turn
        self.id = Vehicle.vehicle_id_counter
        Vehicle.vehicle_id_counter += 1

class TrafficIntersection:
    def __init__(self, arrival_rates, min_green=10, yellow_duration=4):
        self.arrival_rates = arrival_rates
        self.min_green_duration = min_green
        self.yellow_duration = yellow_duration
        self.reset()

    def reset(self):
        """Resets the environment to an initial state."""
        self.lanes = {direction: {turn: deque() for turn in TURNS} for direction in DIRECTIONS}
        self.step_count = 0
        self.total_throughput = 0
        
        self.current_phase = 0  # Start with North-South straight
        self.time_in_phase = 0
        
        # Metrics
        self.total_wait_time = 0
        self.vehicles_passed = 0
        
        return self.get_observation()

    def _generate_vehicles(self):
        """Adds new vehicles to lanes based on arrival rates."""
        for direction in DIRECTIONS:
            for turn in TURNS:
                if random.random() < self.arrival_rates[direction][turn]:
                    vehicle = Vehicle(self.step_count, turn)
                    self.lanes[direction][turn].append(vehicle)

    def step(self, action):
        """
        Executes one time step in the simulation.
        Action: 0 (keep), 1 (switch to NS_STRAIGHT), 2 (switch to NS_LEFT), 
                3 (switch to EW_STRAIGHT), 4 (switch to EW_LEFT)
        """
        # 1. Update phase based on action
        is_yellow_phase = self.current_phase % 2 != 0
        
        if is_yellow_phase:
            # If yellow, must continue until it's over
            if self.time_in_phase >= self.yellow_duration:
                # Yellow phase ends, move to the next requested green phase
                self.current_phase = self.next_green_phase
                self.time_in_phase = 0
        else: # Is a green phase
            # Agent wants to switch and green time is sufficient
            if action != 0 and self.time_in_phase >= self.min_green_duration:
                # Action 1-4 maps to green phases 0, 2, 4, 6
                requested_green_phase = (action - 1) * 2
                if requested_green_phase != self.current_phase:
                    # Start transition to yellow
                    self.current_phase += 1 
                    self.time_in_phase = 0
                    self.next_green_phase = requested_green_phase

        # 2. Process vehicle movement
        self._update_lanes()

        # 3. Add new vehicles
        self._generate_vehicles()

        # 4. Update counters
        self.step_count += 1
        self.time_in_phase += 1
        
        # 5. Calculate reward
        reward = self._calculate_reward()
        
        # 6. Get next state
        observation = self.get_observation()
        
        done = self.step_count >= 1000  # End episode after 1000 steps
        
        return observation, reward, done

    def _update_lanes(self):
        """Moves vehicles through the intersection based on the current signal phase."""
        if self.current_phase not in GREEN_PHASE_LANE_MAP:
            # No movements during yellow phases
            return

        vehicles_cleared_this_step = 0
        
        # One vehicle can pass per green lane per step
        for direction, turn in GREEN_PHASE_LANE_MAP[self.current_phase]:
            if self.lanes[direction][turn]:
                vehicle = self.lanes[direction][turn].popleft()
                self.total_wait_time += self.step_count - vehicle.step_created
                self.vehicles_passed += 1
                vehicles_cleared_this_step += 1
        
        self.total_throughput += vehicles_cleared_this_step

    def _calculate_reward(self):
        """Calculates the reward for the current state."""
        # Negative reward for total queue length
        queue_lengths = [len(self.lanes[d][t]) for d in DIRECTIONS for t in TURNS]
        total_queue = sum(queue_lengths)
        
        # Small penalty for switching phases to encourage efficiency
        phase_switch_penalty = 5 if self.time_in_phase == 1 and self.current_phase % 2 != 0 else 0
        
        return -total_queue - phase_switch_penalty

    def get_observation(self):
        """
        Returns the current state of the environment as a feature vector.
        State vector (28 features):
        - 12 queue lengths
        - 12 max wait times
        - 1-hot encoded current GREEN phase (4 features)
        """
        queue_lengths = [len(self.lanes[d][t]) for d in DIRECTIONS for t in TURNS]
        
        max_wait_times = []
        for d in DIRECTIONS:
            for t in TURNS:
                if self.lanes[d][t]:
                    oldest_vehicle_step = self.lanes[d][t][0].step_created
                    max_wait_times.append(self.step_count - oldest_vehicle_step)
                else:
                    max_wait_times.append(0)

        # 1-hot encode the current GREEN phase (0, 2, 4, 6)
        phase_1_hot = [0, 0, 0, 0]
        green_phase_index = self.current_phase // 2
        if not self.current_phase % 2: # It's a green phase
            phase_1_hot[green_phase_index] = 1

        return np.array(queue_lengths + max_wait_times + phase_1_hot, dtype=np.float32)

    def get_grid_observation(self):
        """Returns a 4x3 grid representing queue lengths for CNN input."""
        grid = np.zeros((4, 3))
        for d_idx, d in enumerate(DIRECTIONS):
            for t_idx, t in enumerate(TURNS):
                grid[d_idx, t_idx] = len(self.lanes[d][t])
        return np.expand_dims(grid, axis=-1) # Add channel dimension

    def get_metrics(self):
        """Returns a dictionary of current environment metrics."""
        metrics = {
            'step': self.step_count,
            'total_queue_length': sum(len(self.lanes[d][t]) for d in DIRECTIONS for t in TURNS),
            'total_throughput': self.total_throughput,
            'current_phase_id': self.current_phase,
            'current_phase_name': PHASES[self.current_phase],
            'time_in_phase': self.time_in_phase,
            'lanes': {}
        }
        for d in DIRECTIONS:
            metrics['lanes'][d] = {}
            for t in TURNS:
                metrics['lanes'][d][t] = {
                    'queue_length': len(self.lanes[d][t]),
                    'queue': [{'id': v.id, 'step_created': v.step_created, 'destination_turn': v.destination_turn} 
                             for v in self.lanes[d][t]]
                }
        return metrics

    def render(self):
        """Prints a text-based representation of the intersection."""
        print(f"--- Step {self.step_count} | Phase: {PHASES[self.current_phase]} ({self.time_in_phase}s) ---")
        for d in DIRECTIONS:
            print(f"{d.capitalize()}:")
            for t in TURNS:
                queue = len(self.lanes[d][t])
                print(f"  - {t.capitalize()}: {queue} vehicles")
        print("-" * 30)
