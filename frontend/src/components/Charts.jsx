# frontend/src/components/Charts.jsx
import React, { useEffect, useState } from 'react';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import './Charts.css';

const Charts = ({ history }) => {
  const [chartData, setChartData] = useState([]);
  const [laneData, setLaneData] = useState([]);

  useEffect(() => {
    if (!history || !history.timestamps) return;

    // Prepare data for charts
    const data = history.timestamps.map((ts, idx) => ({
      step: ts,
      queue: history.queue_lengths[idx],
      throughput: history.throughputs[idx],
      north: history.north_queues[idx],
      south: history.south_queues[idx],
      east: history.east_queues[idx],
      west: history.west_queues[idx]
    }));

    setChartData(data);

    // Prepare lane comparison for latest values
    if (data.length > 0) {
      const latest = data[data.length - 1];
      setLaneData([
        { name: 'North', value: latest.north },
        { name: 'South', value: latest.south },
        { name: 'East', value: latest.east },
        { name: 'West', value: latest.west }
      ]);
    }
  }, [history]);

  const COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'];

  return (
    <div className="charts-container">
      <div className="chart-wrapper">
        <h3>Queue Length Over Time</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#404040" />
            <XAxis dataKey="step" stroke="#999" />
            <YAxis stroke="#999" />
            <Tooltip
              contentStyle={{
                backgroundColor: '#2a2a2a',
                border: '1px solid #0ff',
                borderRadius: '5px'
              }}
              labelStyle={{ color: '#0ff' }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="queue"
              stroke="#ff6b6b"
              dot={false}
              name="Total Queue"
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-wrapper">
        <h3>Throughput (Vehicles Processed)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#404040" />
            <XAxis dataKey="step" stroke="#999" />
            <YAxis stroke="#999" />
            <Tooltip
              contentStyle={{
                backgroundColor: '#2a2a2a',
                border: '1px solid #0f0',
                borderRadius: '5px'
              }}
              labelStyle={{ color: '#0f0' }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="throughput"
              stroke="#4ecdc4"
              dot={false}
              name="Throughput"
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-wrapper">
        <h3>Lane Queue Distribution</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={laneData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#404040" />
            <XAxis dataKey="name" stroke="#999" />
            <YAxis stroke="#999" />
            <Tooltip
              contentStyle={{
                backgroundColor: '#2a2a2a',
                border: '1px solid #45b7d1',
                borderRadius: '5px'
              }}
              labelStyle={{ color: '#45b7d1' }}
            />
            <Bar dataKey="value" fill="#45b7d1" radius={[8, 8, 0, 0]}>
              {laneData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-wrapper">
        <h3>Lane Comparison</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#404040" />
            <XAxis dataKey="step" stroke="#999" />
            <YAxis stroke="#999" />
            <Tooltip
              contentStyle={{
                backgroundColor: '#2a2a2a',
                border: '1px solid #96ceb4',
                borderRadius: '5px'
              }}
              labelStyle={{ color: '#96ceb4' }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="north"
              stroke={COLORS[0]}
              dot={false}
              name="North"
              strokeWidth={2}
            />
            <Line
              type="monotone"
              dataKey="south"
              stroke={COLORS[1]}
              dot={false}
              name="South"
              strokeWidth={2}
            />
            <Line
              type="monotone"
              dataKey="east"
              stroke={COLORS[2]}
              dot={false}
              name="East"
              strokeWidth={2}
            />
            <Line
              type="monotone"
              dataKey="west"
              stroke={COLORS[3]}
              dot={false}
              name="West"
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default Charts;
