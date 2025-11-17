// frontend/src/components/SignalStatus.jsx
import React from 'react';
import './SignalStatus.css';

const SignalStatus = ({ metrics }) => {
  if (!metrics || !metrics.lanes) return null;

  // Helper function to determine signal color for a specific lane
  const getSignalColor = (direction, turn) => {
    const phaseName = metrics.current_phase_name;
    const isYellow = phaseName?.includes('YELLOW');
    
    const greenPhaseMap = {
      'NS_STRAIGHT_GREEN': [['north', 'straight'], ['south', 'straight'], ['north', 'right'], ['south', 'right']],
      'NS_LEFT_GREEN': [['north', 'left'], ['south', 'left']],
      'EW_STRAIGHT_GREEN': [['east', 'straight'], ['west', 'straight'], ['east', 'right'], ['west', 'right']],
      'EW_LEFT_GREEN': [['east', 'left'], ['west', 'left']],
    };

    const currentGreenLanes = greenPhaseMap[phaseName] || [];
    const isGreen = currentGreenLanes.some(([d, t]) => d === direction && t === turn);
    
    if (isGreen) return 'green';
    if (isYellow) {
      const previousPhase = phaseName.replace('YELLOW', 'GREEN');
      const previousGreenLanes = greenPhaseMap[previousPhase] || [];
      if (previousGreenLanes.some(([d, t]) => d === direction && t === turn)) {
        return 'yellow';
      }
    }
    return 'red';
  };

  // Generate reasoning for current phase
  const getPhaseReasoning = () => {
    const phaseName = metrics.current_phase_name;
    const timeInPhase = metrics.time_in_phase;
    const totalQueue = metrics.total_queue_length;

    if (phaseName?.includes('YELLOW')) {
      return `Transitioning phases (safety buffer ${timeInPhase}s). Allowing intersection to clear before switching to next green phase.`;
    }

    const nsQueue = (metrics.lanes.north?.straight?.queue_length || 0) + 
                    (metrics.lanes.south?.straight?.queue_length || 0);
    const ewQueue = (metrics.lanes.east?.straight?.queue_length || 0) + 
                    (metrics.lanes.west?.straight?.queue_length || 0);
    const nsLeftQueue = (metrics.lanes.north?.left?.queue_length || 0) + 
                        (metrics.lanes.south?.left?.queue_length || 0);
    const ewLeftQueue = (metrics.lanes.east?.left?.queue_length || 0) + 
                        (metrics.lanes.west?.left?.queue_length || 0);

    if (phaseName === 'NS_STRAIGHT_GREEN') {
      return `North-South straight traffic has priority (${nsQueue} vehicles). Time: ${timeInPhase}s. Total queue across all lanes: ${totalQueue}.`;
    } else if (phaseName === 'NS_LEFT_GREEN') {
      return `North-South left turns active (${nsLeftQueue} vehicles waiting). Dedicated phase reduces conflicts with oncoming traffic.`;
    } else if (phaseName === 'EW_STRAIGHT_GREEN') {
      return `East-West straight traffic has priority (${ewQueue} vehicles). Time: ${timeInPhase}s. Balancing traffic flow across directions.`;
    } else if (phaseName === 'EW_LEFT_GREEN') {
      return `East-West left turns active (${ewLeftQueue} vehicles waiting). Protected left-turn phase for safety.`;
    }

    return `Active phase: ${phaseName}. Duration: ${timeInPhase}s. Managing ${totalQueue} vehicles total.`;
  };

  // Predict next action
  const getNextAction = () => {
    const phaseName = metrics.current_phase_name;
    const timeInPhase = metrics.time_in_phase;
    const minGreen = 10; // Could pass from config

    if (phaseName?.includes('YELLOW')) {
      const nextPhase = phaseName.includes('NS') ? 'EW_STRAIGHT_GREEN' : 'NS_STRAIGHT_GREEN';
      return `Will switch to ${nextPhase} after transition completes.`;
    }

    const nsQueue = (metrics.lanes.north?.straight?.queue_length || 0) + 
                    (metrics.lanes.south?.straight?.queue_length || 0);
    const ewQueue = (metrics.lanes.east?.straight?.queue_length || 0) + 
                    (metrics.lanes.west?.straight?.queue_length || 0);
    const nsLeftQueue = (metrics.lanes.north?.left?.queue_length || 0) + 
                        (metrics.lanes.south?.left?.queue_length || 0);
    const ewLeftQueue = (metrics.lanes.east?.left?.queue_length || 0) + 
                        (metrics.lanes.west?.left?.queue_length || 0);

    if (timeInPhase < minGreen) {
      return `Maintaining current phase (minimum ${minGreen}s required). ${minGreen - timeInPhase}s remaining before agent can consider switching.`;
    }

    // Agent decision logic
    if (phaseName === 'NS_STRAIGHT_GREEN') {
      if (ewQueue > nsQueue * 1.5) {
        return `Agent considering switch to EW traffic (EW queue: ${ewQueue} vs NS queue: ${nsQueue}). High waiting vehicles on East-West.`;
      } else if (nsLeftQueue > 3) {
        return `May switch to NS left-turn phase next (${nsLeftQueue} vehicles waiting to turn left).`;
      } else {
        return `Continue serving NS straight traffic (efficient throughput, low competing queue: ${ewQueue}).`;
      }
    } else if (phaseName === 'NS_LEFT_GREEN') {
      if (nsLeftQueue === 0) {
        return `Left-turn queue cleared. Will transition to next phase (likely EW straight traffic).`;
      }
      return `Serving NS left turns. Will switch when queue clears or EW pressure builds (EW queue: ${ewQueue}).`;
    } else if (phaseName === 'EW_STRAIGHT_GREEN') {
      if (nsQueue > ewQueue * 1.5) {
        return `Agent considering switch to NS traffic (NS queue: ${nsQueue} vs EW queue: ${ewQueue}). North-South backing up.`;
      } else if (ewLeftQueue > 3) {
        return `May switch to EW left-turn phase next (${ewLeftQueue} vehicles waiting to turn left).`;
      } else {
        return `Continue serving EW straight traffic (efficient flow, NS queue manageable: ${nsQueue}).`;
      }
    } else if (phaseName === 'EW_LEFT_GREEN') {
      if (ewLeftQueue === 0) {
        return `Left-turn queue cleared. Will transition to next phase (likely NS straight traffic).`;
      }
      return `Serving EW left turns. Will switch when queue clears or NS pressure builds (NS queue: ${nsQueue}).`;
    }

    return 'Agent evaluating traffic patterns to optimize flow...';
  };

  const directions = ['north', 'south', 'east', 'west'];
  const turns = ['left', 'straight', 'right'];
  const turnSymbols = { left: '←', straight: '↑', right: '→' };
  const directionEmojis = { north: '⬆️', south: '⬇️', east: '➡️', west: '⬅️' };

  return (
    <div className="signal-status-container">
      <h2>Signal Control Intelligence</h2>
      
      {/* Current Status */}
      <div className="status-section">
        <h3>Current Phase</h3>
        <div className="phase-info">
          <div className="phase-name-large">{metrics.current_phase_name}</div>
          <div className="phase-timer">⏱️ {metrics.time_in_phase}s</div>
        </div>
      </div>

      {/* AI Reasoning */}
      <div className="status-section reasoning-section">
        <h3>🧠 Why This Phase?</h3>
        <p className="reasoning-text">{getPhaseReasoning()}</p>
      </div>

      {/* Next Action */}
      <div className="status-section next-action-section">
        <h3>🎯 Next Action</h3>
        <p className="next-action-text">{getNextAction()}</p>
      </div>

      {/* Detailed Lane Status */}
      <div className="status-section">
        <h3>Lane-by-Lane Status</h3>
        <div className="lanes-status">
          {directions.map(direction => (
            <div key={direction} className="direction-group">
              <h4>{directionEmojis[direction]} {direction.toUpperCase()} Approach</h4>
              <div className="lane-rows">
                {turns.map(turn => {
                  const laneData = metrics.lanes[direction]?.[turn];
                  const queueLength = laneData?.queue_length || 0;
                  const signalColor = getSignalColor(direction, turn);
                  
                  return (
                    <div key={turn} className={`lane-row ${signalColor}-bg`}>
                      <span className="lane-label">
                        {turnSymbols[turn]} {turn}
                      </span>
                      <span className={`signal-indicator signal-${signalColor}`}>
                        ●
                      </span>
                      <span className="queue-count">
                        {queueLength} {queueLength === 1 ? 'vehicle' : 'vehicles'}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Traffic Pressure Indicators */}
      <div className="status-section">
        <h3>📊 Traffic Pressure</h3>
        <div className="pressure-bars">
          <div className="pressure-item">
            <label>North-South</label>
            <div className="pressure-bar-container">
              <div 
                className="pressure-bar ns-bar" 
                style={{ 
                  width: `${Math.min(100, ((metrics.lanes.north?.straight?.queue_length || 0) + 
                          (metrics.lanes.south?.straight?.queue_length || 0)) * 5)}%` 
                }}
              />
            </div>
            <span className="pressure-value">
              {(metrics.lanes.north?.straight?.queue_length || 0) + 
               (metrics.lanes.south?.straight?.queue_length || 0)} vehicles
            </span>
          </div>
          <div className="pressure-item">
            <label>East-West</label>
            <div className="pressure-bar-container">
              <div 
                className="pressure-bar ew-bar" 
                style={{ 
                  width: `${Math.min(100, ((metrics.lanes.east?.straight?.queue_length || 0) + 
                          (metrics.lanes.west?.straight?.queue_length || 0)) * 5)}%` 
                }}
              />
            </div>
            <span className="pressure-value">
              {(metrics.lanes.east?.straight?.queue_length || 0) + 
               (metrics.lanes.west?.straight?.queue_length || 0)} vehicles
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SignalStatus;
