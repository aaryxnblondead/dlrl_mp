// frontend/src/components/Dashboard.jsx
import React, { useState, useEffect, useRef } from 'react';
import Intersection from './Intersection';
import Training from './Training'; // Import the Training component
import './Dashboard.css';

const Dashboard = () => {
  const [config, setConfig] = useState({
    north_traffic: 0.3,
    south_traffic: 0.3,
    east_traffic: 0.25,
    west_traffic: 0.25,
    green_duration: 30,
    use_rl: true
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

          <div className="config-group">
            <label>North Lane Traffic Density</label>
            <input
              type="range"
              min="0.1"
              max="0.5"
              step="0.05"
              value={config.north_traffic}
              onChange={e => handleConfigChange('north_traffic', parseFloat(e.target.value))}
            />
            <span>{config.north_traffic.toFixed(2)}</span>
          </div>

          <div className="config-group">
            <label>South Lane Traffic Density</label>
            <input
              type="range"
              min="0.1"
              max="0.5"
              step="0.05"
              value={config.south_traffic}
              onChange={e => handleConfigChange('south_traffic', parseFloat(e.target.value))}
            />
            <span>{config.south_traffic.toFixed(2)}</span>
          </div>

          <div className="config-group">
            <label>East Lane Traffic Density</label>
            <input
              type="range"
              min="0.1"
              max="0.5"
              step="0.05"
              value={config.east_traffic}
              onChange={e => handleConfigChange('east_traffic', parseFloat(e.target.value))}
            />
            <span>{config.east_traffic.toFixed(2)}</span>
          </div>

          <div className="config-group">
            <label>West Lane Traffic Density</label>
            <input
              type="range"
              min="0.1"
              max="0.5"
              step="0.05"
              value={config.west_traffic}
              onChange={e => handleConfigChange('west_traffic', parseFloat(e.target.value))}
            />
            <span>{config.west_traffic.toFixed(2)}</span>
          </div>

          <div className="config-group">
            <label>Green Light Duration (steps)</label>
            <input
              type="range"
              min="10"
              max="60"
              step="5"
              value={config.green_duration}
              onChange={e => handleConfigChange('green_duration', parseInt(e.target.value))}
            />
            <span>{config.green_duration}</span>
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
            <div className="metrics-grid">
              <div className="metric-card">
                <label>Total Queue Length</label>
                <span className="metric-value">{metrics.total_queue_length}</span>
              </div>
              <div className="metric-card">
                <label>Total Throughput</label>
                <span className="metric-value">{metrics.total_throughput}</span>
              </div>
              <div className="metric-card">
                <label>Current Phase</label>
                <span className="metric-value">{metrics.current_phase}</span>
              </div>
              <div className="metric-card">
                <label>Phase Time</label>
                <span className="metric-value">{metrics.time_in_phase}s</span>
              </div>

              <div className="metric-card">
                <label>North Queue</label>
                <span className="metric-value">{metrics.north.queue_length}</span>
              </div>
              <div className="metric-card">
                <label>South Queue</label>
                <span className="metric-value">{metrics.south.queue_length}</span>
              </div>
              <div className="metric-card">
                <label>East Queue</label>
                <span className="metric-value">{metrics.east.queue_length}</span>
              </div>
              <div className="metric-card">
                <label>West Queue</label>
                <span className="metric-value">{metrics.west.queue_length}</span>
              </div>
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
