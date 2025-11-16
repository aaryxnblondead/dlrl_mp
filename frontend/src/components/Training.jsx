// frontend/src/components/Training.jsx
import React, { useState, useEffect } from 'react';
import './Training.css';

const Training = () => {
  const [status, setStatus] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [totalConfigs, setTotalConfigs] = useState(0);
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);

  const API_BASE = 'http://localhost:5000/api';

  const pollStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/training/status`);
      const data = await response.json();

      if (response.ok) {
        setStatus(data.status);
        setProgress(data.progress);
        setTotalConfigs(data.total_configs);

        // Dynamically update results as they come in
        if (data.results_count > results.length) {
          fetchResults();
        }

        if (data.status === 'completed') {
          fetchResults();
        }
      } else {
        setError(data.message || 'Failed to get status.');
      }
    } catch (err) {
      setError('Error connecting to the server.');
    }
  };

  const fetchResults = async () => {
    try {
      const response = await fetch(`${API_BASE}/training/results`);
      const data = await response.json();
      if (response.ok) {
        // Sort results by average reward, descending
        const sortedResults = data.results.sort((a, b) => b.avg_reward - a.avg_reward);
        setResults(sortedResults);
      } else {
        setError(data.message || 'Failed to fetch results.');
      }
    } catch (err) {
      setError('Error fetching results.');
    }
  };

  useEffect(() => {
    let interval;
    if (status === 'running') {
      interval = setInterval(pollStatus, 2000); // Poll every 2 seconds
    } else {
      // Initial load
      pollStatus();
    }
    return () => clearInterval(interval);
  }, [status]);

  const handleStartTraining = async () => {
    setError(null);
    setResults([]);
    try {
      const response = await fetch(`${API_BASE}/training/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({}), // Send an empty JSON object
      });
      const data = await response.json();
      if (response.ok) {
        setStatus('running');
        setTotalConfigs(data.total_configurations);
        setProgress(0);
      } else {
        setError(data.message || 'Failed to start training.');
      }
    } catch (err) {
      setError('Could not start training process.');
    }
  };

  const renderProgressBar = () => {
    if (status !== 'running' || totalConfigs === 0) return null;
    const percentage = Math.round((progress / totalConfigs) * 100);
    return (
      <div className="progress-bar-container">
        <div className="progress-bar" style={{ width: `${percentage}%` }}>
          {percentage}%
        </div>
        <span>{`Processing ${progress} of ${totalConfigs} configurations...`}</span>
      </div>
    );
  };

  const renderResultsTable = () => {
    if (results.length === 0) return null;
    return (
      <div className="results-container">
        <h3>Training Results Document</h3>
        <p>Showing all combinations sorted by the highest reward (best performance).</p>
        <table className="results-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Avg Reward</th>
              <th>Avg Queue</th>
              <th>N Traffic</th>
              <th>S Traffic</th>
              <th>E Traffic</th>
              <th>W Traffic</th>
              <th>Green Time</th>
            </tr>
          </thead>
          <tbody>
            {results.map((res, index) => (
              <tr key={res.config_id} className={res.error ? 'error-result' : (index === 0 ? 'best-result' : '')}>
                <td>{index + 1}</td>
                {res.error ? (
                  <td colSpan="2">Error: {res.error}</td>
                ) : (
                  <>
                    <td>{res.avg_reward.toFixed(2)}</td>
                    <td>{res.avg_queue_length.toFixed(2)}</td>
                  </>
                )}
                <td>{res.config.north_traffic}</td>
                <td>{res.config.south_traffic}</td>
                <td>{res.config.east_traffic}</td>
                <td>{res.config.west_traffic}</td>
                <td>{res.config.green_duration}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="training-panel">
      <h2>Train Agent</h2>
      <p>Run the agent through all possible combinations of traffic density and signal timings to find the optimal strategy.</p>
      
      {error && <div className="error-message">{error}</div>}

      {status === 'idle' && (
        <button className="btn btn-primary" onClick={handleStartTraining}>
          Start Full Training
        </button>
      )}

      {status === 'running' && (
        <div className="status-running">
          <p>Training in progress...</p>
          {renderProgressBar()}
          <button className="btn" disabled>Training...</button>
        </div>
      )}

      {status === 'completed' && (
        <div className="status-completed">
          <p>Training complete! Results are displayed below.</p>
          <button className="btn btn-primary" onClick={handleStartTraining}>
            Run Training Again
          </button>
        </div>
      )}

      {renderResultsTable()}
    </div>
  );
};

export default Training;
