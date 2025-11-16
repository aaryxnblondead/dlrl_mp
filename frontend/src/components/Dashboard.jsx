// frontend/src/components/Dashboard.jsx
import React, { useState, useEffect, useRef } from 'react';
import Intersection from './Intersection';
import Training from './Training'; // Import the Training component
import './Dashboard.css';

const ConfigSlider = ({ label, value, onChange }) => (
  <div className="config-group">
    <label>{label}</label>
    <input
      type="range"
      min="0.01"
      max="0.5"
      step="0.01"
      value={value}
      onChange={e => onChange(parseFloat(e.target.value))}
    />
    <span>{value.toFixed(2)}</span>
  </div>
);

const Dashboard = () => {
  const [config, setConfig] = useState({
    north_left: 0.1, north_straight: 0.2, north_right: 0.1,
    south_left: 0.1, south_straight: 0.2, south_right: 0.1,
    east_left: 0.08, east_straight: 0.15, east_right: 0.08,
    west_left: 0.08, west_straight: 0.15, west_right: 0.08,
    min_green: 10,
    yellow_duration: 4,
    use_rl: true,
    use_cnn: false,
  });

  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState(null);
  const [running, setRunning] = useState(false);
  const [comparison, setComparison] = useState(null);
  const stepIntervalRef = useRef(null);

  const API_BASE = 'http://localhost:5000/api';

  const handleInitialize = async () => {
    try {
      const response = await fetch(`${API_BASE}/initialize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      const result = await response.json();
      console.log(result.message);
      // After initializing, get the initial state
      const metricsResponse = await fetch(`${API_BASE}/metrics`);
      if (metricsResponse.ok) {
        const metricsData = await metricsResponse.json();
        setMetrics(metricsData.current);
      }
    } catch (error) {
      console.error('Error initializing:', error);
    }
  };

  const handleStart = async () => {
    await handleInitialize();
    setRunning(true);
  };

  const handleStop = () => {
    setRunning(false);
  };

  const handleStep = async () => {
    if (!running) return;
    try {
      const response = await fetch(`${API_BASE}/step`, {
        method: 'POST'
      });
      const result = await response.json();
      setMetrics(result);

      // Optionally update history less frequently
      if (result.step % 10 === 0) {
        const historyResponse = await fetch(`${API_BASE}/history?limit=200`);
        const historyData = await historyResponse.json();
        setHistory(historyData);
      }
    } catch (error) {
      console.error('Error stepping:', error);
      setRunning(false); // Stop on error
    }
  };

  useEffect(() => {
    if (running) {
      stepIntervalRef.current = setInterval(handleStep, 200); // Adjust for desired speed
    } else {
      if (stepIntervalRef.current) {
        clearInterval(stepIntervalRef.current);
      }
    }
    return () => {
      if (stepIntervalRef.current) {
        clearInterval(stepIntervalRef.current);
      }
    };
  }, [running]);


  const handleCompare = async () => {
    try {
      const response = await fetch(`${API_BASE}/compare`, {
        method: 'POST'
      });
      const result = await response.json();
      setComparison(result);
    } catch (error) {
      console.error('Error comparing:', error);
    }
  };

  const handleConfigChange = (field, value) => {
    setConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>Smart Traffic Signal Optimizer</h1>
        <p>RL-based traffic management system</p>
      </header>

      <div className="dashboard-container">
        {/* Configuration Panel */}
        <div className="panel config-panel">
          <h2>Configuration</h2>

          <div className="config-group-grid">
            <h3>North Approach</h3>
            <ConfigSlider label="Left" value={config.north_left} onChange={v => handleConfigChange('north_left', v)} />
            <ConfigSlider label="Straight" value={config.north_straight} onChange={v => handleConfigChange('north_straight', v)} />
            <ConfigSlider label="Right" value={config.north_right} onChange={v => handleConfigChange('north_right', v)} />
          </div>
          <div className="config-group-grid">
            <h3>South Approach</h3>
            <ConfigSlider label="Left" value={config.south_left} onChange={v => handleConfigChange('south_left', v)} />
            <ConfigSlider label="Straight" value={config.south_straight} onChange={v => handleConfigChange('south_straight', v)} />
            <ConfigSlider label="Right" value={config.south_right} onChange={v => handleConfigChange('south_right', v)} />
          </div>
          <div className="config-group-grid">
            <h3>East Approach</h3>
            <ConfigSlider label="Left" value={config.east_left} onChange={v => handleConfigChange('east_left', v)} />
            <ConfigSlider label="Straight" value={config.east_straight} onChange={v => handleConfigChange('east_straight', v)} />
            <ConfigSlider label="Right" value={config.east_right} onChange={v => handleConfigChange('east_right', v)} />
          </div>
          <div className="config-group-grid">
            <h3>West Approach</h3>
            <ConfigSlider label="Left" value={config.west_left} onChange={v => handleConfigChange('west_left', v)} />
            <ConfigSlider label="Straight" value={config.west_straight} onChange={v => handleConfigChange('west_straight', v)} />
            <ConfigSlider label="Right" value={config.west_right} onChange={v => handleConfigChange('west_right', v)} />
          </div>

          <div className="config-group">
            <label>Min Green Time (steps)</label>
            <input
              type="range" min="5" max="30" step="1"
              value={config.min_green}
              onChange={e => handleConfigChange('min_green', parseInt(e.target.value))}
            />
            <span>{config.min_green}</span>
          </div>

          <div className="config-group">
            <label>
              <input
                type="checkbox"
                checked={config.use_rl}
                onChange={e => handleConfigChange('use_rl', e.target.checked)}
              />
              Use RL Agent
            </label>
          </div>
          <div className="config-group">
            <label>
              <input
                type="checkbox"
                checked={config.use_cnn}
                onChange={e => handleConfigChange('use_cnn', e.target.checked)}
              />
              Use CNN Model
            </label>
          </div>

          <div className="button-group">
            {!running ? (
              <button className="btn btn-primary" onClick={handleStart}>
                Start Simulation
              </button>
            ) : (
              <button className="btn btn-danger" onClick={handleStop}>
                Stop Simulation
              </button>
            )}
            <button className="btn btn-secondary" onClick={handleCompare}>
              Compare Algorithms
            </button>
          </div>
        </div>

        {/* Visualization */}
        <div className="panel visualization-panel">
          <h2>Intersection Simulation</h2>
          <Intersection metrics={metrics} />
        </div>

        {/* Metrics Panel */}
        <div className="panel metrics-panel">
          <h2>Real-time Metrics</h2>
          {metrics && (
            <div className="metrics-grid-condensed">
              <div className="metric-card">
                <label>Total Queue</label>
                <span className="metric-value">{metrics.total_queue_length}</span>
              </div>
              <div className="metric-card">
                <label>Throughput</label>
                <span className="metric-value">{metrics.total_throughput}</span>
              </div>
              <div className="metric-card">
                <label>Phase</label>
                <span className="metric-value phase-name">{metrics.current_phase_name}</span>
              </div>
              <div className="metric-card">
                <label>Phase Time</label>
                <span className="metric-value">{metrics.time_in_phase}s</span>
              </div>
              {/* Detailed lane metrics can be added here if needed */}
            </div>
          )}
        </div>

        {/* Comparison Results */}
        {comparison && (
          <div className="panel comparison-panel">
            <h2>Algorithm Comparison (500 steps)</h2>
            <div className="comparison-grid">
              <div className="comparison-card">
                <h3>RL Agent</h3>
                <p>Avg Queue: {comparison.rl.avg_queue.toFixed(2)}</p>
                <p>Throughput: {comparison.rl.avg_throughput.toFixed(2)}</p>
                <p>Total Reward: {comparison.rl.total_reward.toFixed(2)}</p>
              </div>
              <div className="comparison-card">
                <h3>Fixed Timing</h3>
                <p>Avg Queue: {comparison.fixed.avg_queue.toFixed(2)}</p>
                <p>Throughput: {comparison.fixed.avg_throughput.toFixed(2)}</p>
                <p>Total Reward: {comparison.fixed.total_reward.toFixed(2)}</p>
              </div>
            </div>
          </div>
        )}

        {/* Training Panel */}
        <div className="panel training-panel-container">
          <Training />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
