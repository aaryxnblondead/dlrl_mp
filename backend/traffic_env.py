# traffic_env.py
import numpy as np
from collections import deque
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

class SignalPhase(Enum):
    """Traffic signal phases"""
    NORTH_SOUTH = 0
    EAST_WEST = 1

@dataclass
class LaneMetrics:
    """Metrics for a single lane"""
    vehicle_count: int
    queue_length: int
    avg_wait_time: float
    throughput: int

class TrafficLane:
    """Simulates a single traffic lane"""
    def __init__(self, capacity: int = 30, arrival_rate: float = 0.3):
        self.capacity = capacity
        self.arrival_rate = arrival_rate
        self.vehicles = deque()  # (arrival_time, wait_time)
        self.max_wait = 0
        self.throughput = 0
        self.total_wait_time = 0
        self.vehicles_processed = 0
        
    def step(self, is_green: bool, time_step: int):
        """Simulate one time step in the lane"""
        # New arrivals
        if np.random.random() < self.arrival_rate and len(self.vehicles) < self.capacity:
            self.vehicles.append({'arrival': time_step, 'wait_time': 0})
        
        # Update wait times
        for vehicle in self.vehicles:
            vehicle['wait_time'] += 1
        
        # Process vehicles if light is green
        if is_green and len(self.vehicles) > 0:
            vehicle = self.vehicles.popleft()
            wait = vehicle['wait_time']
            self.total_wait_time += wait
            self.max_wait = max(self.max_wait, wait)
            self.vehicles_processed += 1
            self.throughput += 1
        
        return {
            'queue_length': len(self.vehicles),
            'max_wait': self.max_wait if self.vehicles_processed > 0 else 0
        }
    
    def get_metrics(self) -> LaneMetrics:
        """Get current lane metrics"""
        avg_wait = self.total_wait_time / max(1, self.vehicles_processed)
        return LaneMetrics(
            vehicle_count=len(self.vehicles),
            queue_length=len(self.vehicles),
            avg_wait_time=avg_wait,
            throughput=self.throughput
        )
    
    def reset(self):
        """Reset lane state"""
        self.vehicles.clear()
        self.throughput = 0
        self.total_wait_time = 0
        self.vehicles_processed = 0
        self.max_wait = 0

class TrafficIntersection:
    """Simulates a 4-way traffic intersection with 4 lanes"""
    def __init__(self, 
                 arrival_rates: Optional[Dict[str, float]] = None,
                 green_duration: int = 30,
                 yellow_duration: int = 5):
        """
        Args:
            arrival_rates: Dict with keys 'north', 'south', 'east', 'west'
            green_duration: Duration of green light in time steps
            yellow_duration: Duration of yellow light in time steps
        """
        self.arrival_rates = arrival_rates or {
            'north': 0.3, 'south': 0.3, 'east': 0.25, 'west': 0.25
        }
        
        self.lanes = {
            'north': TrafficLane(arrival_rate=self.arrival_rates['north']),
            'south': TrafficLane(arrival_rate=self.arrival_rates['south']),
            'east': TrafficLane(arrival_rate=self.arrival_rates['east']),
            'west': TrafficLane(arrival_rate=self.arrival_rates['west'])
        }
        
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.current_phase = SignalPhase.NORTH_SOUTH
        self.time_in_phase = 0
        self.current_time = 0
        self.episode_total_wait = 0
        self.episode_vehicles_processed = 0
        
    def get_observation(self) -> np.ndarray:
        """
        Get state observation as 4D array: [N, S, E, W]
        Each dimension: [queue_length, max_wait, phase_time, total_vehicles_waiting]
        """
        observation = []
        total_waiting = 0
        
        for direction in ['north', 'south', 'east', 'west']:
            lane = self.lanes[direction]
            observation.extend([
                len(lane.vehicles),  # Queue length
                lane.max_wait if lane.vehicles_processed > 0 else 0,  # Max wait
                self.time_in_phase / self.green_duration,  # Normalized phase time
            ])
            total_waiting += len(lane.vehicles)
        
        observation.append(total_waiting / 120)  # Normalized total waiting
        
        return np.array(observation, dtype=np.float32)
    
    def get_grid_observation(self) -> np.ndarray:
        """
        Get observation as grid representation for CNN processing
        Shape: (4, 4) representing intersection state
        """
        grid = np.zeros((4, 4), dtype=np.float32)
        
        # North lane
        grid[0, 1:3] = min(len(self.lanes['north'].vehicles) / 10, 1.0)
        # South lane
        grid[3, 1:3] = min(len(self.lanes['south'].vehicles) / 10, 1.0)
        # East lane
        grid[1:3, 3] = min(len(self.lanes['east'].vehicles) / 10, 1.0)
        # West lane
        grid[1:3, 0] = min(len(self.lanes['west'].vehicles) / 10, 1.0)
        
        # Center represents signal state
        grid[1:3, 1:3] = 1.0 if self.current_phase == SignalPhase.NORTH_SOUTH else 0.5
        
        return grid
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute one time step
        
        Args:
            action: 0 = keep current phase, 1 = switch phase
        
        Returns:
            observation, reward, done
        """
        # Handle phase switching
        if action == 1 and self.time_in_phase >= self.green_duration - self.yellow_duration:
            self.current_phase = SignalPhase(1 - self.current_phase.value)
            self.time_in_phase = 0
        
        self.time_in_phase += 1
        
        # Determine which lanes have green light
        is_ns_green = (self.current_phase == SignalPhase.NORTH_SOUTH)
        is_ew_green = (self.current_phase == SignalPhase.EAST_WEST)
        
        # Step each lane
        for direction, is_green in [('north', is_ns_green), ('south', is_ns_green),
                                    ('east', is_ew_green), ('west', is_ew_green)]:
            self.lanes[direction].step(is_green, self.current_time)
        
        # Calculate reward (minimize total wait time)
        total_wait = sum(len(lane.vehicles) for lane in self.lanes.values())
        throughput = sum(lane.throughput for lane in self.lanes.values())
        
        # Reward: negative total queue length + positive throughput
        reward = -total_wait * 0.5 + throughput * 0.1
        
        self.episode_total_wait += total_wait
        self.current_time += 1
        
        return self.get_observation(), reward, False
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        for lane in self.lanes.values():
            lane.reset()
        
        self.current_phase = SignalPhase.NORTH_SOUTH
        self.time_in_phase = 0
        self.current_time = 0
        self.episode_total_wait = 0
        
        return self.get_observation()
    
    def get_metrics(self) -> Dict:
        """Get current intersection metrics"""
        metrics = {}
        total_wait = 0
        total_throughput = 0
        
        for direction in ['north', 'south', 'east', 'west']:
            lane_metrics = self.lanes[direction].get_metrics()
            metrics[direction] = {
                'queue_length': lane_metrics.queue_length,
                'avg_wait_time': lane_metrics.avg_wait_time,
                'throughput': lane_metrics.throughput
            }
            total_wait += lane_metrics.queue_length
            total_throughput += lane_metrics.throughput
        
        metrics['total_queue_length'] = total_wait
        metrics['total_throughput'] = total_throughput
        metrics['current_phase'] = self.current_phase.name
        metrics['time_in_phase'] = self.time_in_phase
        
        return metrics
    
    def update_arrival_rates(self, arrival_rates: Dict[str, float]):
        """Update traffic arrival rates (for user configuration)"""
        self.arrival_rates = arrival_rates
        for direction, rate in arrival_rates.items():
            self.lanes[direction].arrival_rate = rate
