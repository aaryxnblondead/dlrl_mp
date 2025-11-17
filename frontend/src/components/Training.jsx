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

        // Only fetch results when training is completed
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
        const sortedResults = data.results.sort((a, b) => (b.avg_reward || -Infinity) - (a.avg_reward || -Infinity));
        setResults(sortedResults);
        setError(null); // Clear any previous errors
      } else {
        // Don't show error if training just hasn't been started yet
        if (status !== 'idle') {
          setError(data.message || 'Failed to fetch results.');
        }
      }
    } catch (err) {
      // Only show error if we're actually running/completed
      if (status !== 'idle') {
        setError('Error fetching results.');
      }
    }
  };

  useEffect(() => {
    let interval;
    if (status === 'running') {
      interval = setInterval(pollStatus, 2000); // Poll every 2 seconds
    } else if (status === 'idle') {
      // Initial load - just check status, don't fetch results
      pollStatus();
    } else if (status === 'completed') {
      // When completed, fetch results once
      fetchResults();
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
    
    // Check if using comprehensive mode (has per-lane configs) or quick mode (has per-direction configs)
    const isComprehensive = results.length > 0 && results[0].config && 'north_left' in results[0].config;
    
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
              {isComprehensive ? (
                <>
                  <th>N-L</th>
                  <th>N-S</th>
                  <th>N-R</th>
                  <th>S-L</th>
                  <th>S-S</th>
                  <th>S-R</th>
                  <th>E-L</th>
                  <th>E-S</th>
                  <th>E-R</th>
                  <th>W-L</th>
                  <th>W-S</th>
                  <th>W-R</th>
                </>
              ) : (
                <>
                  <th>N Traffic</th>
                  <th>S Traffic</th>
                  <th>E Traffic</th>
                  <th>W Traffic</th>
                </>
              )}
              <th>Min Green</th>
            </tr>
          </thead>
          <tbody>
            {results.map((res, index) => (
              <tr key={res.config_id || index} className={res.error ? 'error-result' : (index === 0 ? 'best-result' : '')}>
                <td>{index + 1}</td>
                {res.error ? (
                  <td colSpan={isComprehensive ? 14 : 6}>Error: {res.error}</td>
                ) : (
                  <>
                    <td>{res.avg_reward.toFixed(2)}</td>
                    <td>{res.avg_queue_length.toFixed(2)}</td>
                    {isComprehensive ? (
                      <>
                        <td>{res.config.north_left}</td>
                        <td>{res.config.north_straight}</td>
                        <td>{res.config.north_right}</td>
                        <td>{res.config.south_left}</td>
                        <td>{res.config.south_straight}</td>
                        <td>{res.config.south_right}</td>
                        <td>{res.config.east_left}</td>
                        <td>{res.config.east_straight}</td>
                        <td>{res.config.east_right}</td>
                        <td>{res.config.west_left}</td>
                        <td>{res.config.west_straight}</td>
                        <td>{res.config.west_right}</td>
                      </>
                    ) : (
                      <>
                        <td>{res.config.north_traffic}</td>
                        <td>{res.config.south_traffic}</td>
                        <td>{res.config.east_traffic}</td>
                        <td>{res.config.west_traffic}</td>
                      </>
                    )}
                    <td>{res.config.min_green}</td>
                  </>
                )}
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
