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
        
        # Track previous state for corrective rewards
        self.prev_approach_queues = {d: 0 for d in DIRECTIONS}
        self.prev_lane_queues = {f"{d}_{t}": 0 for d in DIRECTIONS for t in TURNS}
        self.high_pressure_approaches = set()  # Track which approaches need urgent attention
        self.gridlock_lanes = set()  # Track lanes in gridlock (>20 vehicles)
        
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
        Action: 0 (keep), 1 (NS_STRAIGHT), 2 (NS_LEFT), 3 (NS_RIGHT),
                4 (EW_STRAIGHT), 5 (EW_LEFT), 6 (EW_RIGHT)
        Note: NS_RIGHT and EW_RIGHT map to NS_STRAIGHT and EW_STRAIGHT phases
        since right turns are allowed with straight greens.
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
                # Action mapping:
                # 1: NS_STRAIGHT (phase 0), 2: NS_LEFT (phase 2), 3: NS_RIGHT (phase 0)
                # 4: EW_STRAIGHT (phase 4), 5: EW_LEFT (phase 6), 6: EW_RIGHT (phase 4)
                action_to_phase = {1: 0, 2: 2, 3: 0, 4: 4, 5: 6, 6: 4}
                requested_green_phase = action_to_phase.get(action, self.current_phase)
                
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
        """Calculates reward prioritizing traffic pressure relief, gridlock prevention, and flow efficiency.
        Each approach road (north, south, east, west) is treated independently.
        Includes real-time corrective rewards for actively reducing pressure after penalties."""
        queue_lengths = [len(self.lanes[d][t]) for d in DIRECTIONS for t in TURNS]
        total_queue = sum(queue_lengths)
        max_queue = max(queue_lengths) if queue_lengths else 0
        
        # Track current state
        current_approach_queues = {d: sum([len(self.lanes[d][t]) for t in TURNS]) for d in DIRECTIONS}
        current_lane_queues = {f"{d}_{t}": len(self.lanes[d][t]) for d in DIRECTIONS for t in TURNS}
        
        # CORE PRINCIPLE: LOW QUEUES = HIGH REWARD
        # Reward for keeping total queue low
        if total_queue < 20:
            low_queue_reward = (20 - total_queue) ** 2 * 10
        else:
            low_queue_reward = 0
        
        # PRIORITY 1: PREVENT BUILDUPS - Start penalizing EARLY
        gridlock_penalty = 0
        new_gridlock_lanes = set()
        for d in DIRECTIONS:
            for t in TURNS:
                lane_key = f"{d}_{t}"
                q = current_lane_queues[lane_key]
                if q > 20:
                    gridlock_penalty -= 100 * ((q - 20) ** 2)  # Catastrophic
                    new_gridlock_lanes.add(lane_key)
                elif q > 15:
                    gridlock_penalty -= 50 * ((q - 15) ** 2)  # Heavy penalty
                elif q > 10:
                    gridlock_penalty -= 20 * (q - 10)  # Moderate penalty
                elif q < 3:
                    gridlock_penalty += 10 * (3 - q)  # Reward for low queues
        
        # CORRECTIVE REWARD 1: Massive bonus for reducing gridlock lanes
        gridlock_reduction_reward = 0
        for lane_key in self.gridlock_lanes:
            prev_q = self.prev_lane_queues.get(lane_key, 0)
            curr_q = current_lane_queues[lane_key]
            if prev_q > 20 and curr_q <= 20:
                # Successfully cleared a gridlocked lane below threshold!
                gridlock_reduction_reward += 500
            elif prev_q > 20 and curr_q < prev_q:
                # Making progress on gridlocked lane
                reduction = prev_q - curr_q
                gridlock_reduction_reward += reduction * 50
        
        # PRIORITY 2: REWARD LOW APPROACH QUEUES
        approach_rewards = 0
        approach_pressure_penalty = 0
        new_high_pressure_approaches = set()
        for direction in DIRECTIONS:
            approach_total = current_approach_queues[direction]
            
            if approach_total == 0:
                approach_rewards += 50  # Perfect!
            elif approach_total < 5:
                approach_rewards += (5 - approach_total) * 10  # Very good
            elif approach_total < 10:
                approach_rewards += (10 - approach_total) * 2  # Acceptable
            elif approach_total < 15:
                approach_pressure_penalty -= (approach_total - 10) * 10  # Building
            elif approach_total < 20:
                approach_pressure_penalty -= ((approach_total - 15) ** 2) * 5
                new_high_pressure_approaches.add(direction)
            else:
                approach_pressure_penalty -= ((approach_total - 20) ** 2) * 10  # Critical
                new_high_pressure_approaches.add(direction)
        
        # CORRECTIVE REWARD 2: Strong bonus for actively clearing high-pressure approaches
        pressure_reduction_reward = 0
        for direction in self.high_pressure_approaches:
            prev_total = self.prev_approach_queues.get(direction, 0)
            curr_total = current_approach_queues[direction]
            
            if prev_total > 15 and curr_total < prev_total:
                # Actively reducing pressure on a high-pressure approach
                reduction = prev_total - curr_total
                # Higher bonus for bigger reductions
                pressure_reduction_reward += reduction * 30
                
            if prev_total > 20 and curr_total <= 15:
                # Brought a severely congested approach back to manageable levels
                pressure_reduction_reward += 300
        
        # CORRECTIVE REWARD 3: Strong bonus for serving the highest-queue approach
        if current_approach_queues:
            # Rank all approaches by queue size
            sorted_approaches = sorted(current_approach_queues.items(), key=lambda x: x[1], reverse=True)
            highest_queue_direction = sorted_approaches[0][0]
            highest_queue_count = sorted_approaches[0][1]
            second_highest_count = sorted_approaches[1][1] if len(sorted_approaches) > 1 else 0
            
            current_green_phase = self.current_phase if self.current_phase % 2 == 0 else None
            if current_green_phase is not None and current_green_phase in GREEN_PHASE_LANE_MAP:
                serving_directions = set([d for d, t in GREEN_PHASE_LANE_MAP[current_green_phase]])
                
                if highest_queue_direction in serving_directions:
                    # Reward for serving highest queue, but LESS reward if queue is already big
                    if highest_queue_count > 15:
                        pressure_reduction_reward += 30  # Too late, small reward
                    elif highest_queue_count > 8:
                        pressure_reduction_reward += 80  # Better
                    else:
                        pressure_reduction_reward += 200  # Great! Caught it early
                else:
                    # NOT serving highest queue approach - PENALTY
                    if highest_queue_count > 8:  # Reduced from 15 - switch earlier!
                        pressure_reduction_reward -= highest_queue_count * 20
                    elif highest_queue_count > 5:
                        pressure_reduction_reward -= highest_queue_count * 10
        
        # PRIORITY 3: PER-LANE PRESSURE within each approach
        lane_pressure_penalty = 0
        for q in queue_lengths:
            if q > 10:
                lane_pressure_penalty -= (q - 10) * 3
            if q > 15:
                lane_pressure_penalty -= ((q - 15) ** 2) * 2
        
        # CORRECTIVE REWARD 4: Bonus for any lane-level reduction
        lane_reduction_reward = 0
        for lane_key, curr_q in current_lane_queues.items():
            prev_q = self.prev_lane_queues.get(lane_key, 0)
            if prev_q > 10 and curr_q < prev_q:
                # Reward proportional to reduction for pressured lanes
                reduction = prev_q - curr_q
                lane_reduction_reward += reduction * 8
        
        # PRIORITY 4: FLOW EFFICIENCY - Reward clearing vehicles aggressively
        throughput_reward = self.vehicles_passed * 3.0
        
        # Per-approach clearing bonus: prioritize clearing from higher-queue approaches
        approach_clearing_bonus = 0
        if current_approach_queues:
            # Rank approaches by queue size
            sorted_approaches = sorted(current_approach_queues.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (direction, approach_total) in enumerate(sorted_approaches):
                if approach_total > 5 and self.vehicles_passed > 0:
                    # Higher bonus for clearing from higher-ranked (more congested) approaches
                    rank_multiplier = (4 - rank) ** 2  # 1st=16x, 2nd=9x, 3rd=4x, 4th=1x
                    approach_clearing_bonus += (approach_total / 6) * rank_multiplier
        
        # Base queue penalty - STRONG penalty for ANY queue
        queue_penalty = -total_queue * 8
        
        # Critical wait time penalty - start earlier at 15 steps
        critical_wait_penalty = 0
        for d in DIRECTIONS:
            for t in TURNS:
                if self.lanes[d][t]:
                    oldest_vehicle_step = self.lanes[d][t][0].step_created
                    wait_time = self.step_count - oldest_vehicle_step
                    if wait_time > 15:  # Earlier threshold
                        critical_wait_penalty -= ((wait_time - 15) ** 2) * 3
        
        # Phase stability: balanced penalty to avoid thrashing while allowing priority switches
        switch_penalty = -5 if self.time_in_phase == 1 and self.current_phase % 2 != 0 else 0
        
        # SMART SWITCHING BONUS: Reward switching TO serve higher-priority approaches
        smart_switch_bonus = 0
        if self.time_in_phase == 1 and self.current_phase % 2 == 0 and current_approach_queues:
            # Just switched to a new green phase
            current_green_phase = self.current_phase
            if current_green_phase in GREEN_PHASE_LANE_MAP:
                serving_directions = set([d for d, t in GREEN_PHASE_LANE_MAP[current_green_phase]])
                
                # Check if we're now serving a high-priority approach
                sorted_approaches = sorted(current_approach_queues.items(), key=lambda x: x[1], reverse=True)
                highest_queue_direction = sorted_approaches[0][0]
                highest_queue_count = sorted_approaches[0][1]
                
                # Reward switching to serve building queues
                for direction in serving_directions:
                    queue_size = current_approach_queues.get(direction, 0)
                    if queue_size > 5:
                        # HIGHER reward for catching queues EARLY
                        if queue_size < 10:
                            smart_switch_bonus = 200  # Excellent early switch!
                        elif queue_size < 15:
                            smart_switch_bonus = 120  # Good switch
                        else:
                            smart_switch_bonus = 50   # Late but necessary
                        switch_penalty = 0  # Cancel switch penalty
                        break
        
        # Individual approach balance: prevent extreme disparities between ANY approaches
        approach_totals = list(current_approach_queues.values())
        max_approach = max(approach_totals) if approach_totals else 0
        min_approach = min(approach_totals) if approach_totals else 0
        approach_disparity = max_approach - min_approach
        
        disparity_penalty = 0
        if approach_disparity > 15:
            disparity_penalty -= (approach_disparity ** 1.5) * 0.5
        
        # Turn-specific handling within each approach
        turn_imbalance_penalty = 0
        for d in DIRECTIONS:
            left_q = len(self.lanes[d]['left'])
            straight_q = len(self.lanes[d]['straight'])
            right_q = len(self.lanes[d]['right'])
            max_turn_q = max(left_q, straight_q, right_q)
            min_turn_q = min(left_q, straight_q, right_q)
            if max_turn_q > 15 and (max_turn_q - min_turn_q) > 10:
                turn_imbalance_penalty -= (max_turn_q - min_turn_q) * 2
        
        # Update tracking for next step
        self.prev_approach_queues = current_approach_queues.copy()
        self.prev_lane_queues = current_lane_queues.copy()
        self.high_pressure_approaches = new_high_pressure_approaches.copy()
        self.gridlock_lanes = new_gridlock_lanes.copy()
        
        total_reward = (low_queue_reward + gridlock_penalty + gridlock_reduction_reward + 
                       approach_rewards + approach_pressure_penalty + pressure_reduction_reward +
                       lane_pressure_penalty + lane_reduction_reward +
                       throughput_reward + approach_clearing_bonus + queue_penalty + 
                       critical_wait_penalty + switch_penalty + smart_switch_bonus +
                       disparity_penalty + turn_imbalance_penalty)
        
        return total_reward

    def get_observation(self):
        """
        Returns enhanced state with pressure indicators for better decision-making.
        State vector (44 features):
        - 12 queue lengths (raw counts)
        - 12 normalized queue lengths (0-1 scale, capped at 50)
        - 12 max wait times
        - 4 per-direction total queues
        - 1-hot encoded current GREEN phase (4 features)
        """
        queue_lengths = [len(self.lanes[d][t]) for d in DIRECTIONS for t in TURNS]
        
        # Normalized queue lengths to help network learn thresholds
        normalized_queues = [min(q / 50.0, 1.0) for q in queue_lengths]
        
        max_wait_times = []
        for d in DIRECTIONS:
            for t in TURNS:
                if self.lanes[d][t]:
                    oldest_vehicle_step = self.lanes[d][t][0].step_created
                    max_wait_times.append(self.step_count - oldest_vehicle_step)
                else:
                    max_wait_times.append(0)
        
        # Per-direction pressure (total vehicles waiting in each approach)
        direction_pressures = []
        for d in DIRECTIONS:
            total = sum([len(self.lanes[d][t]) for t in TURNS])
            direction_pressures.append(total)

        # 1-hot encode the current GREEN phase (0, 2, 4, 6)
        phase_1_hot = [0, 0, 0, 0]
        green_phase_index = self.current_phase // 2
        if not self.current_phase % 2: # It's a green phase
            phase_1_hot[green_phase_index] = 1

        return np.array(queue_lengths + normalized_queues + max_wait_times + 
                       direction_pressures + phase_1_hot, dtype=np.float32)

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
