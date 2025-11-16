// frontend/src/components/Intersection.jsx
import React, { useEffect, useRef } from 'react';
import './Intersection.css';

const Intersection = ({ metrics }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !metrics) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const laneWidth = 80;

    // Clear canvas
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, width, height);

    // Draw road
    ctx.fillStyle = '#333';

    // Horizontal road
    ctx.fillRect(0, centerY - laneWidth / 2, width, laneWidth);

    // Vertical road
    ctx.fillRect(centerX - laneWidth / 2, 0, laneWidth, height);

    // Draw lanes
    ctx.strokeStyle = '#ffff00';
    ctx.lineWidth = 2;
    ctx.setLineDash([10, 10]);

    // Lane markings
    ctx.beginPath();
    ctx.moveTo(centerX - 20, 0);
    ctx.lineTo(centerX - 20, height);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(centerX + 20, 0);
    ctx.lineTo(centerX + 20, height);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, centerY - 20);
    ctx.lineTo(width, centerY - 20);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, centerY + 20);
    ctx.lineTo(width, centerY + 20);
    ctx.stroke();

    ctx.setLineDash([]);

    // Draw traffic lights
    const drawTrafficLight = (x, y, isGreen) => {
      const lightRadius = 15;

      ctx.fillStyle = '#333';
      ctx.fillRect(x - 20, y - 35, 40, 60);

      ctx.fillStyle = isGreen ? '#00ff00' : '#ff0000';
      ctx.beginPath();
      ctx.arc(x, y - 15, lightRadius, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#ffaa00';
      ctx.beginPath();
      ctx.arc(x, y, lightRadius, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#ff0000';
      ctx.beginPath();
      ctx.arc(x, y + 15, lightRadius, 0, Math.PI * 2);
      ctx.fill();
    };

    // Check signal phase
    const isNSGreen = metrics.current_phase === 'NORTH_SOUTH';

    drawTrafficLight(centerX - 50, centerY - laneWidth / 2 - 20, isNSGreen);
    drawTrafficLight(centerX + 50, centerY + laneWidth / 2 + 20, isNSGreen);
    drawTrafficLight(centerX - laneWidth / 2 - 20, centerY - 50, !isNSGreen);
    drawTrafficLight(centerX + laneWidth / 2 + 20, centerY + 50, !isNSGreen);

    // Draw vehicles as rectangles
    const drawVehicles = (x, y, count, direction) => {
      const vehicleSize = 15;
      const spacing = 20;
      const maxVehicles = 5;

      const displayCount = Math.min(count, maxVehicles);

      for (let i = 0; i < displayCount; i++) {
        ctx.fillStyle = '#ff0000';
        let vx, vy, vw, vh;

        if (direction === 'north') {
          vx = x - vehicleSize / 2;
          vy = y + i * spacing;
          vw = vehicleSize;
          vh = vehicleSize;
        } else if (direction === 'south') {
          vx = x - vehicleSize / 2;
          vy = y - i * spacing;
          vw = vehicleSize;
          vh = vehicleSize;
        } else if (direction === 'east') {
          vx = x - i * spacing;
          vy = y - vehicleSize / 2;
          vw = vehicleSize;
          vh = vehicleSize;
        } else if (direction === 'west') {
          vx = x + i * spacing;
          vy = y - vehicleSize / 2;
          vw = vehicleSize;
          vh = vehicleSize;
        }

        ctx.fillRect(vx, vy, vw, vh);
      }
    };

    // Draw vehicles for each lane
    if (metrics.north && metrics.north.queue_length) {
      drawVehicles(centerX + 20, centerY - laneWidth / 2 - 30, metrics.north.queue_length, 'north');
    }
    if (metrics.south && metrics.south.queue_length) {
      drawVehicles(centerX - 20, centerY + laneWidth / 2 + 30, metrics.south.queue_length, 'south');
    }
    if (metrics.east && metrics.east.queue_length) {
      drawVehicles(centerX + laneWidth / 2 + 30, centerY + 20, metrics.east.queue_length, 'east');
    }
    if (metrics.west && metrics.west.queue_length) {
      drawVehicles(centerX - laneWidth / 2 - 30, centerY - 20, metrics.west.queue_length, 'west');
    }
  }, [metrics]);

  return <canvas ref={canvasRef} width={600} height={600} className="intersection-canvas" />;
};

export default Intersection;
